#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "IcpOptimizer.h"

namespace py = pybind11;

using namespace std;
using namespace Eigen;

std::pair<py::array_t<double>, py::array_t<double>> sparse_icp(py::array_t<double> source_pcd, py::array_t<double> target_pcd, size_t kNormals, int nbIterations, int nbIterationsIn, double mu, int nbIterShrink, double p, int method, bool verbose)
{
    py::buffer_info buf_info_source = source_pcd.request();
    py::buffer_info buf_info_target = target_pcd.request();

    if (buf_info_source.shape[1] != 3 || buf_info_target.shape[1] != 3)
    {
        throw std::runtime_error("Number of columns must be three!");
    }

    Matrix<double, Dynamic, 3> pointCloudOne = Map<Matrix<double, Dynamic, 3>>(reinterpret_cast<double *>(buf_info_source.ptr), buf_info_source.shape[0], buf_info_source.shape[1]);
    Matrix<double, Dynamic, 3> pointCloudTwo = Map<Matrix<double, Dynamic, 3>>(reinterpret_cast<double *>(buf_info_target.ptr), buf_info_target.shape[0], buf_info_target.shape[1]);

    // Creating an IcpOptimizer in order to perform the sparse icp
    IcpOptimizer myIcpOptimizer(pointCloudOne, pointCloudTwo, kNormals, nbIterations, nbIterationsIn, mu, nbIterShrink, p, static_cast<IcpMethod>(method), verbose);

    // Perform ICP
    int hasIcpFailed = myIcpOptimizer.performSparceICP();
    if (hasIcpFailed)
    {
        throw std::invalid_argument("Failed to load the point clouds. Check the paths.");
    }
    RigidTransfo result = myIcpOptimizer.getComputedTransfo();

    RotMatrix rot = std::get<0>(result);
    TransMatrix trs = std::get<1>(result);

    py::array_t<double> rot_array = py::array_t<double>(rot.size(), rot.data());
    py::array_t<double> trs_array = py::array_t<double>(trs.size(), trs.data());
    return std::make_pair(rot_array, trs_array);
}

PYBIND11_MODULE(sparseIcp, m)
{
    m.def("sparse_icp", &sparse_icp, "sparse icp");
}
