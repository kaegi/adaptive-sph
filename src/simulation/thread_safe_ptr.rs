use std::ops::Deref;

pub struct ThreadSafeConstPtr<T> {
    ptr: *const T,
}
unsafe impl<T> Sync for ThreadSafeConstPtr<T> {}
unsafe impl<T> Send for ThreadSafeConstPtr<T> {}
impl<T> ThreadSafeConstPtr<T> {
    pub fn new(ptr: *const T) -> ThreadSafeConstPtr<T> {
        ThreadSafeConstPtr { ptr }
    }
}
impl<T> Deref for ThreadSafeConstPtr<T> {
    type Target = *const T;
    fn deref(&self) -> &Self::Target {
        &self.ptr
    }
}
impl<T> Copy for ThreadSafeConstPtr<T> {}
impl<T> Clone for ThreadSafeConstPtr<T> {
    fn clone(&self) -> Self {
        ThreadSafeConstPtr { ptr: self.ptr }
    }
}

#[derive(Copy, Clone)]
pub struct ThreadSafeMutPtr<T> {
    ptr: *mut T,
}
unsafe impl<T> Sync for ThreadSafeMutPtr<T> {}
unsafe impl<T> Send for ThreadSafeMutPtr<T> {}
impl<T> ThreadSafeMutPtr<T> {
    pub fn new(ptr: *mut T) -> ThreadSafeMutPtr<T> {
        ThreadSafeMutPtr { ptr }
    }
}
impl<T> Deref for ThreadSafeMutPtr<T> {
    type Target = *mut T;
    fn deref(&self) -> &Self::Target {
        &self.ptr
    }
}
