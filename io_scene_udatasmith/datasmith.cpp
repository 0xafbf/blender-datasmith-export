

#include <Python.h>


static PyObject *
load_datasmith(PyObject *self, PyObject *args)
{
    
    return Py_BuildValue("s", "datasmith");
}

static PyMethodDef module_methods[] = {
	{"load_datasmith", (PyCFunction)load_datasmith, METH_O, nullptr},
	{nullptr, nullptr, 0, nullptr}
};

static PyModuleDef datasmith_module = {
	PyModuleDef_HEAD_INIT,
	"datasmith",
	"module to interface with datasmith exporter",
	0,
	module_methods
};

PyMODINIT_FUNC PyInit_datasmith()
{
	return PyModule_Create(&datasmith_module);
}