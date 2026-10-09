//! tinyllm_rs (fixture ds.92): a CPython extension written against the
//! stable ABI by hand, so the harness's tl-py build (cdylib, abi3, PYO3_PYTHON,
//! macOS dynamic_lookup, copy to tinyllm_rs.so) is exercised offline.
#![allow(non_snake_case, non_camel_case_types, static_mut_refs)]
use std::os::raw::{c_char, c_int, c_long, c_void};
use std::ptr;

#[repr(C)]
pub struct PyObject {
    ob_refcnt: isize,
    ob_type: *mut c_void,
}
type PyCFunction = unsafe extern "C" fn(*mut PyObject, *mut PyObject) -> *mut PyObject;
#[repr(C)]
struct PyMethodDef {
    ml_name: *const c_char,
    ml_meth: Option<PyCFunction>,
    ml_flags: c_int,
    ml_doc: *const c_char,
}
#[repr(C)]
struct PyModuleDef_Base {
    ob_base: PyObject,
    m_init: Option<unsafe extern "C" fn() -> *mut PyObject>,
    m_index: isize,
    m_copy: *mut PyObject,
}
#[repr(C)]
struct PyModuleDef {
    m_base: PyModuleDef_Base,
    m_name: *const c_char,
    m_doc: *const c_char,
    m_size: isize,
    m_methods: *mut PyMethodDef,
    m_slots: *mut c_void,
    m_traverse: *mut c_void,
    m_clear: *mut c_void,
    m_free: *mut c_void,
}

extern "C" {
    fn PyModule_Create2(def: *mut PyModuleDef, apiver: c_int) -> *mut PyObject;
    fn PyArg_ParseTuple(args: *mut PyObject, fmt: *const c_char, ...) -> c_int;
    fn PyLong_FromLong(v: c_long) -> *mut PyObject;
    fn PyFloat_FromDouble(v: f64) -> *mut PyObject;
    fn PySequence_Size(o: *mut PyObject) -> isize;
    fn PySequence_GetItem(o: *mut PyObject, i: isize) -> *mut PyObject;
    fn PyFloat_AsDouble(o: *mut PyObject) -> f64;
    fn Py_DecRef(o: *mut PyObject);
    fn PyErr_Occurred() -> *mut PyObject;
}

const METH_VARARGS: c_int = 1;

/// add(a, b) -> a + b, two Python ints.
unsafe extern "C" fn add(_m: *mut PyObject, args: *mut PyObject) -> *mut PyObject {
    // SOLUTION-BEGIN ds.92
    let (mut a, mut b): (c_long, c_long) = (0, 0);
    if PyArg_ParseTuple(args, c"ll".as_ptr(), &mut a as *mut c_long, &mut b as *mut c_long) == 0 {
        return ptr::null_mut();
    }
    PyLong_FromLong(a + b)
    // SOLUTION-END
}

/// kahan(seq) -> the compensated sum of a sequence of floats (ds.90).
unsafe extern "C" fn kahan(_m: *mut PyObject, args: *mut PyObject) -> *mut PyObject {
    // SOLUTION-BEGIN ds.92
    let mut seq: *mut PyObject = ptr::null_mut();
    if PyArg_ParseTuple(args, c"O".as_ptr(), &mut seq as *mut *mut PyObject) == 0 {
        return ptr::null_mut();
    }
    let n = PySequence_Size(seq);
    if n < 0 {
        return ptr::null_mut();
    }
    let mut xs = Vec::with_capacity(n as usize);
    for i in 0..n {
        let it = PySequence_GetItem(seq, i);
        if it.is_null() {
            return ptr::null_mut();
        }
        let v = PyFloat_AsDouble(it);
        Py_DecRef(it);
        if v == -1.0 && !PyErr_Occurred().is_null() {
            return ptr::null_mut();
        }
        xs.push(v);
    }
    PyFloat_FromDouble(tl_demo::acc::kahan_sum(&xs))
    // SOLUTION-END
}

static mut METHODS: [PyMethodDef; 3] = [
    PyMethodDef { ml_name: c"add".as_ptr(), ml_meth: Some(add), ml_flags: METH_VARARGS, ml_doc: ptr::null() },
    PyMethodDef { ml_name: c"kahan".as_ptr(), ml_meth: Some(kahan), ml_flags: METH_VARARGS, ml_doc: ptr::null() },
    PyMethodDef { ml_name: ptr::null(), ml_meth: None, ml_flags: 0, ml_doc: ptr::null() },
];

static mut MODULE: PyModuleDef = PyModuleDef {
    m_base: PyModuleDef_Base {
        ob_base: PyObject { ob_refcnt: 1, ob_type: ptr::null_mut() },
        m_init: None,
        m_index: 0,
        m_copy: ptr::null_mut(),
    },
    m_name: c"tinyllm_rs".as_ptr(),
    m_doc: ptr::null(),
    m_size: -1,
    m_methods: ptr::null_mut(),
    m_slots: ptr::null_mut(),
    m_traverse: ptr::null_mut(),
    m_clear: ptr::null_mut(),
    m_free: ptr::null_mut(),
};

/// The module init the interpreter looks up by name (glue, never stubbed).
#[no_mangle]
pub unsafe extern "C" fn PyInit_tinyllm_rs() -> *mut PyObject {
    MODULE.m_methods = ptr::addr_of_mut!(METHODS) as *mut PyMethodDef;
    PyModule_Create2(ptr::addr_of_mut!(MODULE), 3) // 3 = PYTHON_ABI_VERSION (abi3)
}
