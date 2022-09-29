#pragma once
#include "open3d/Open3D.h"
#include "mesh/mesh.hpp"

namespace mesh
{
    template<typename PointT>
    void show_mesh(const Mesh<PointT>& mesh)
    {
        std::shared_ptr<open3d::geometry::TriangleMesh> me 
            = std::make_shared<open3d::geometry::TriangleMesh>();
        me->vertices_ = mesh.points_;
        me->triangles_ = mesh.triangles_;
        open3d::visualization::DrawGeometries({me});
        open3d::io::WriteTriangleMesh("./data/mesh.ply", *me);
    }
}