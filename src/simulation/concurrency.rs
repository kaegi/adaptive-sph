pub use internal::*;

#[cfg(target_arch = "wasm32")]
mod internal {

    pub fn par_iter_reduce2<
        T1: Send + Sync,
        T2: Send + Sync,
        F: Fn(usize, &mut T1, &mut T2) -> X + Send + Sync,
        X: Send,
        C: Fn(X, X) -> X + Send + Sync,
        I: Fn() -> X + Send + Sync,
    >(
        arr1: &mut [T1],
        arr2: &mut [T2],
        identity: I,
        combine: C,
        f: F,
    ) -> X {
        arr1.into_iter()
            .zip(arr2.into_iter())
            .enumerate()
            .map(|(i, (a, b))| f(i, a, b))
            .reduce(|acc, value| combine(acc, value))
            .unwrap_or_else(identity)
    }

    pub fn into_par_iter<T>(
        v: impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T>>,
    ) -> impl Iterator<Item = T> {
        v.into_iter()
    }

    pub fn par_iter_mut0<F: Fn(usize) + Send + Sync>(n: usize, f: F) {
        (0..n).into_iter().for_each(|idx| {
            f(idx);
        });
    }

    pub fn par_iter_mut1<T1: Send + Sync, F: Fn(usize, &mut T1) + Send + Sync>(arr1: &mut [T1], f: F) {
        arr1.into_iter().enumerate().for_each(|(idx, v1)| {
            f(idx, v1);
        });
        // arr1.into_iter().enumerate().for_each(|(idx, v1)| {
        //     f(idx, v1);
        // });
    }

    pub fn par_iter_mut2<T1: Send + Sync, T2: Send + Sync, F: Fn(usize, &mut T1, &mut T2) + Send + Sync>(
        arr1: &mut [T1],
        arr2: &mut [T2],
        f: F,
    ) {
        arr1.into_iter()
            .zip(arr2.into_iter())
            .enumerate()
            .for_each(|(idx, (v1, v2))| {
                f(idx, v1, v2);
            });
    }

    pub fn par_iter_mut3<
        T1: Send + Sync,
        T2: Send + Sync,
        T3: Send + Sync,
        F: Fn(usize, &mut T1, &mut T2, &mut T3) + Send + Sync,
    >(
        arr1: &mut [T1],
        arr2: &mut [T2],
        arr3: &mut [T3],
        f: F,
    ) {
        arr1.into_iter()
            .zip(arr2.into_iter())
            .zip(arr3.into_iter())
            .enumerate()
            .for_each(|(idx, ((v1, v2), v3))| {
                f(idx, v1, v2, v3);
            });
    }

    pub fn par_iter_mut4<
        T1: Send + Sync,
        T2: Send + Sync,
        T3: Send + Sync,
        T4: Send + Sync,
        F: Fn(usize, &mut T1, &mut T2, &mut T3, &mut T4) + Send + Sync,
    >(
        arr1: &mut [T1],
        arr2: &mut [T2],
        arr3: &mut [T3],
        arr4: &mut [T4],
        f: F,
    ) {
        arr1.into_iter()
            .zip(arr2.into_iter())
            .zip(arr3.into_iter())
            .zip(arr4.into_iter())
            .enumerate()
            .for_each(|(idx, (((v1, v2), v3), v4))| {
                f(idx, v1, v2, v3, v4);
            });
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod internal {
    use rayon::prelude::*;

    pub fn into_par_iter<T>(v: impl IntoParallelIterator<Item = T>) -> impl ParallelIterator<Item = T> {
        v.into_par_iter()
    }

    pub fn par_iter_reduce2<
        T1: Send + Sync,
        T2: Send + Sync,
        F: Fn(usize, &mut T1, &mut T2) -> X + Send + Sync,
        X: Send,
        C: Fn(X, X) -> X + Send + Sync,
        I: Fn() -> X + Send + Sync,
    >(
        arr1: &mut [T1],
        arr2: &mut [T2],
        identity: I,
        combine: C,
        f: F,
    ) -> X {
        (arr1, arr2)
            .into_par_iter()
            .enumerate()
            .map(|(i, (a, b))| f(i, a, b))
            .reduce(identity, combine)
    }

    pub fn par_iter_mut0<F: Fn(usize) + Send + Sync>(n: usize, f: F) {
        (0..n).into_par_iter().for_each(|idx| {
            f(idx);
        });
    }

    pub fn par_iter_mut1<T1: Send + Sync, F: Fn(usize, &mut T1) + Send + Sync>(arr1: &mut [T1], f: F) {
        arr1.into_par_iter().enumerate().for_each(|(idx, v1)| {
            f(idx, v1);
        });
        // arr1.into_iter().enumerate().for_each(|(idx, v1)| {
        //     f(idx, v1);
        // });
    }

    pub fn par_iter_mut2<T1: Send + Sync, T2: Send + Sync, F: Fn(usize, &mut T1, &mut T2) + Send + Sync>(
        arr1: &mut [T1],
        arr2: &mut [T2],
        f: F,
    ) {
        arr1.into_par_iter()
            .zip(arr2.into_par_iter())
            .enumerate()
            .for_each(|(idx, (v1, v2))| {
                f(idx, v1, v2);
            });
    }

    pub fn par_iter_mut3<
        T1: Send + Sync,
        T2: Send + Sync,
        T3: Send + Sync,
        F: Fn(usize, &mut T1, &mut T2, &mut T3) + Send + Sync,
    >(
        arr1: &mut [T1],
        arr2: &mut [T2],
        arr3: &mut [T3],
        f: F,
    ) {
        arr1.into_par_iter()
            .zip(arr2.into_par_iter())
            .zip(arr3.into_par_iter())
            .enumerate()
            .for_each(|(idx, ((v1, v2), v3))| {
                f(idx, v1, v2, v3);
            });
    }

    pub fn par_iter_mut4<
        T1: Send + Sync,
        T2: Send + Sync,
        T3: Send + Sync,
        T4: Send + Sync,
        F: Fn(usize, &mut T1, &mut T2, &mut T3, &mut T4) + Send + Sync,
    >(
        arr1: &mut [T1],
        arr2: &mut [T2],
        arr3: &mut [T3],
        arr4: &mut [T4],
        f: F,
    ) {
        arr1.into_par_iter()
            .zip(arr2.into_par_iter())
            .zip(arr3.into_par_iter())
            .zip(arr4.into_par_iter())
            .enumerate()
            .for_each(|(idx, (((v1, v2), v3), v4))| {
                f(idx, v1, v2, v3, v4);
            });
    }
}

/*
#[allow(dead_code)]
pub fn for_par_crossbeam<'scope, T: Send + Sync + 'scope, F: Fn(usize, &mut T) + Send + Sync + 'scope>(
    elements: &mut Vec<T>,
    f: F,
) {
    let elements_ptr = &ThreadSafeMutPtr::new(elements.as_mut_ptr());
    let elements_len = elements.len();
    let num_threads = 4;
    let f2 = &f;
    thread::scope(|s| {
        for i in 0..num_threads {
            s.spawn(move |_| {
                let from = (elements_len * i) / num_threads;
                let to = (elements_len * (i + 1)) / num_threads;
                for id in from..to {
                    unsafe {
                        f2(id, &mut *elements_ptr.add(id));
                    }
                }
            });
        }
    })
    .unwrap();
}
*/
