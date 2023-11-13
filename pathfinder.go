// Copyright 2023 Frederik Zipp. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package pathfind finds the shortest path between two points
// constrained by a set of polygons.
package pathfind

import (
	"image"
	"math"
	"sync"

	"github.com/fzipp/geom"
	"github.com/superboySB/astar"
	"github.com/superboySB/pathfind/internal/poly"
)

// A Pathfinder is created and initialized with a set of polygons via
// NewPathfinder. Its Path method finds the shortest path between two points
// in this polygon set.
type Pathfinder struct {
	polygons        [][]image.Point
	polygonSet      poly.PolygonSet
	concaveVertices []image.Point
	visibilityGraph graph[image.Point]
}

// NewPathfinder creates a Pathfinder instance and initializes it with a set of
// polygons.
//
// A polygon is represented by a slice of points, i.e. []image.Point, describing
// the vertices of the polygon. Thus [][]image.Point is a slice of polygons,
// i.e. the set of polygons.
//
// Each polygon in the polygon set designates either an area that is accessible
// for path finding or a hole inside such an area, i.e. an obstacle. Nested
// polygons alternate between accessible area and inaccessible hole:
//   - Polygons at the first level are area polygons.
//   - Polygons contained inside an area polygon are holes.
//   - Polygons contained inside a hole are area polygons again.
func NewPathfinder(polygons [][]image.Point) *Pathfinder {
	polygonSet := convert(polygons, func(ps []image.Point) poly.Polygon {
		return ps2vs(ps)
	})
	return &Pathfinder{
		polygons:        polygons,
		polygonSet:      polygonSet,
		concaveVertices: concaveVertices(polygonSet),
	}
}

// VisibilityGraph returns the calculated visibility graph from the last Path
// call. It is only available after Path was called, otherwise nil.
func (p *Pathfinder) VisibilityGraph() map[image.Point][]image.Point {
	return p.visibilityGraph
}

// Path finds the shortest path from start to dest within the bounds of the
// polygons the Pathfinder was initialized with.
// If dest is outside the polygon set it will be clamped to the nearest
// polygon edge.
// The function returns nil if no path exists because start is outside
// the polygon set.
func (p *Pathfinder) Path(start, dest image.Point) []image.Point {
	d := p2v(dest)
	if !p.polygonSet.Contains(d) {
		dest = ensureInside(p.polygonSet, v2p(p.polygonSet.ClosestPt(d)))
	}
	graphVertices := append(p.concaveVertices, start, dest)
	p.visibilityGraph = visibilityGraph(p.polygonSet, graphVertices)
	return astar.FindPath[image.Point](p.visibilityGraph, start, dest, nodeDist, nodeDist)
}

func ensureInside(ps poly.PolygonSet, pt image.Point) image.Point {
	if ps.Contains(p2v(pt)) {
		return pt
	}
adjustment:
	for dx := -1; dx <= 1; dx++ {
		for dy := -1; dy <= 1; dy++ {
			if dx == 0 && dy == 0 {
				continue
			}
			npt := pt.Add(image.Point{X: dx, Y: dy})
			if ps.Contains(p2v(npt)) {
				pt = npt
				break adjustment
			}
		}
	}
	return pt
}

func concaveVertices(ps poly.PolygonSet) []image.Point {
	var vs []image.Point
	for i, p := range ps {
		t := concave
		if isHole(ps, i) {
			t = convex
		}
		vs = append(vs, verticesOfType(p, t)...)
	}
	return vs
}

func isHole(ps poly.PolygonSet, i int) bool {
	hole := false
	for j, p := range ps {
		if i != j && p.Contains(ps[i][0], false) {
			hole = !hole
		}
	}
	return hole
}

type vertexType int

const (
	concave = vertexType(iota)
	convex
)

func verticesOfType(p poly.Polygon, t vertexType) []image.Point {
	var vs []image.Point
	for i, v := range p {
		isConcave := p.IsConcaveAt(i)
		if (t == concave && isConcave) || (t == convex && !isConcave) {
			vs = append(vs, v2p(v))
		}
	}
	return vs
}

// 原有实现
// func visibilityGraph(ps poly.PolygonSet, points []image.Point) graph[image.Point] {
// 	vis := make(graph[image.Point])
// 	for i, a := range points {
// 		for j, b := range points {
// 			if i == j {
// 				continue
// 			}
// 			if inLineOfSight(ps, p2v(a), p2v(b)) {
// 				vis.link(a, b)
// 			}
// 		}
// 	}
// 	return vis
// }

// 目前实现
func visibilityGraph(ps poly.PolygonSet, points []image.Point) graph[image.Point] {
	vis := make(graph[image.Point], len(points))
	var wg sync.WaitGroup
	var mutex sync.Mutex // 定义互斥锁

	// 根据您的 CPU 逻辑核心数量来设置并发量
	semaphore := make(chan struct{}, 32)

	for i, a := range points {
		wg.Add(1)
		semaphore <- struct{}{}
		go func(a image.Point, i int) {
			defer wg.Done()
			localGraph := make(graph[image.Point], len(points)) // 为每个 goroutine 预分配内存
			for j, b := range points {
				if i == j {
					continue
				}
				if inLineOfSight(ps, p2v(a), p2v(b)) {
					localGraph.link(a, b) // 不需要锁
				}
			}
			// 合并局部图到全局图
			mutex.Lock()
			for node, neighbours := range localGraph {
				vis[node] = append(vis[node], neighbours...)
			}
			mutex.Unlock()
			<-semaphore
		}(a, i)
	}

	wg.Wait()
	close(semaphore)
	return vis
}

func inLineOfSight(ps poly.PolygonSet, start, end geom.Vec2) bool {
	lineOfSight := poly.LineSeg{A: start, B: end}
	for _, p := range ps {
		if p.IsCrossedBy(lineOfSight) {
			return false
		}
	}
	return ps.Contains(lineOfSight.Middle())
}

// nodeDist is the cost function for the A* algorithm. The visibility graph has
// 2d points as nodes, so we calculate the Euclidean distance.
func nodeDist(a, b image.Point) float64 {
	c := a.Sub(b)
	return math.Sqrt(float64(c.X*c.X + c.Y*c.Y))
}
