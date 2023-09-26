#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "CVC_cluster.h"

namespace py = pybind11;

PYBIND11_MODULE(pycluster, m) {

    m.doc() = "Python CVC_cluster";
    m.attr("__version__") = 1;

    
    py::class_<cvc::PointAPR>(m,"PointAPR")
        .def(py::init<>())
        .def_readwrite("azimuth", &cvc::PointAPR::azimuth)
        .def_readwrite("range", &cvc::PointAPR::range)
        .def_readwrite("polar_angle", &cvc::PointAPR::polar_angle); 

    py::class_<cvc::PointXYZ>(m,"PointXYZ")
        .def(py::init<float,float,float>())
        .def_readwrite("x", &cvc::PointXYZ::x)
        .def_readwrite("y", &cvc::PointXYZ::y)
        .def_readwrite("z", &cvc::PointXYZ::z); 

    py::class_<cvc::Voxel>(m,"Voxel")
        .def(py::init<>())
        .def_readwrite("haspoint", &cvc::Voxel::haspoint)
        .def_readwrite("cluster", &cvc::Voxel::cluster)
        .def_readwrite("index", &cvc::Voxel::index); 

    py::class_<cvc::CVC>(m, "CVC_cluster")
        .def(py::init<>())
        .def(py::init<std::vector<float>&>())
        .def("calculateAPR",       &cvc::CVC::calculateAPR)
        .def("build_hash_table",       &cvc::CVC::build_hash_table)
        .def("find_neighbors",       &cvc::CVC::find_neighbors)
        .def("most_frequent_value",       &cvc::CVC::most_frequent_value)
        .def("mergeClusters",       &cvc::CVC::mergeClusters)
        .def("cluster",       &cvc::CVC::cluster); 
        //.def("process",       &cvc::CVC::process); 
}