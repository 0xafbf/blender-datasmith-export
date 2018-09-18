

#include <Python.h>


static PyObject *
load_datasmith(PyObject *self, PyObject *args)
{
    
    return Py_BuildValue("s", "datasmith");
}

static PyObject* PyInit_datasmith()
{
    PyMethodDef MethodDef;
    MethodDef.ml_name = "load_datasmith";
    MethodDef.ml_meth = load_datasmith;
    

    PyModuleDef ModuleDef;
    ModuleDef.m_base = PyModuleDef_HEAD_INIT;
    ModuleDef.m_name = "datasmith";
    ModuleDef.m_doc = "binary interface to datasmith";
    ModuleDef.m_methods = 
    PyModule_Create(&ModuleDef);
}