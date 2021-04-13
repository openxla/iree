vm.module @list_ops {

  //===--------------------------------------------------------------------===//
  // vm.list.* with I8 types
  //===--------------------------------------------------------------------===//

  vm.export @test_i8
  vm.func @test_i8() {
    %c42 = vm.const.i32 42 : i32
    %c100 = vm.const.i32 100 : i32
    %c0 = vm.const.i32 0 : i32
    %list = vm.list.alloc %c42 : (i32) -> !vm.list<i8>
    vm.list.reserve %list, %c100 : (!vm.list<i8>, i32)
    %sz = vm.list.size %list : (!vm.list<i8>) -> i32
    %sz_dno = iree.do_not_optimize(%sz) : i32
    vm.check.eq %sz_dno, %c0, "list<i8>.empty.size()=0" : i32
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.list.* with I16 types
  //===--------------------------------------------------------------------===//

  vm.export @test_i16
  vm.func @test_i16() {
    %c0 = vm.const.i32 0 : i32
    %c1 = vm.const.i32 1 : i32
    %c27 = vm.const.i32 27 : i32
    %list = vm.list.alloc %c1 : (i32) -> !vm.list<i16>
    vm.list.resize %list, %c1 : (!vm.list<i16>, i32)
    vm.list.set.i32 %list, %c0, %c27 : (!vm.list<i16>, i32, i32)
    %v = vm.list.get.i32 %list, %c0 : (!vm.list<i16>, i32) -> i32
    vm.check.eq %v, %c27, "list<i16>.empty.set(0, 27).get(0)=27" : i32
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.list.* with I32 types
  //===--------------------------------------------------------------------===//

  vm.export @test_i32
  vm.func @test_i32() {
    %c42 = vm.const.i32 42 : i32
    %list = vm.list.alloc %c42 : (i32) -> !vm.list<i32>
    %sz = vm.list.size %list : (!vm.list<i32>) -> i32
    %c100 = vm.const.i32 100 : i32
    %c101 = vm.const.i32 101 : i32
    vm.list.resize %list, %c101 : (!vm.list<i32>, i32)
    vm.list.set.i32 %list, %c100, %c42 : (!vm.list<i32>, i32, i32)
    %v = vm.list.get.i32 %list, %c100 : (!vm.list<i32>, i32) -> i32
    vm.check.eq %v, %c42, "list<i32>.empty.set(100, 42).get(100)=42" : i32
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.list.* with I64 types
  //===--------------------------------------------------------------------===//

  vm.export @test_i64
  vm.func @test_i64() {
    %capacity = vm.const.i32 42 : i32
    %index = vm.const.i32 41 : i32
    %max_int_plus_1 = vm.const.i64 2147483648 : i64
    %list = vm.list.alloc %capacity : (i32) -> !vm.list<i64>
    %sz = vm.list.size %list : (!vm.list<i64>) -> i32
    vm.list.resize %list, %capacity : (!vm.list<i64>, i32)
    vm.list.set.i64 %list, %index, %max_int_plus_1 : (!vm.list<i64>, i32, i64)
    %v = vm.list.get.i64 %list, %index : (!vm.list<i64>, i32) -> i64
    vm.check.eq %v, %max_int_plus_1, "list<i64>.empty.set(41, MAX_INT_PLUS_1).get(41)=MAX_INT_PLUS_1" : i64
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.list.* with ref types
  //===--------------------------------------------------------------------===//

  vm.export @test_ref
  vm.func @test_ref() {
    // TODO(benvanik): test vm.list with ref types.
    vm.return
  }
<<<<<<< HEAD
=======

  //===--------------------------------------------------------------------===//
  // vm.list.* with variant types
  //===--------------------------------------------------------------------===//

  vm.export @test_variant
  vm.func @test_variant() {
    // TODO(benvanik): test vm.list with variant types.
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.list.* with control flow
  //===--------------------------------------------------------------------===//

  vm.export @test_control_flow
  vm.func @test_control_flow() {
    // TODO(simon-camp): Implement type converter in the Vm to EmitC conversion. 
  //   %c0 = vm.const.i32 0 : i32
  //   %c1 = vm.const.i32 1 : i32
  //   %c2 = vm.const.i32 2 : i32
  //   %c1_dno = iree.do_not_optimize(%c1) : i32
  //   %list = vm.list.alloc %c2 : (i32) -> !vm.list<i32>
  //   vm.list.resize %list, %c2 : (!vm.list<i32>, i32)
  //   vm.list.set.i32 %list, %c0, %c1 : (!vm.list<i32>, i32, i32)
  //   vm.list.set.i32 %list, %c1, %c2 : (!vm.list<i32>, i32, i32)
  //   vm.cond_br %c1_dno, ^bb1(%list : !vm.list<i32>), ^bb2
  // ^bb1(%list_0 : !vm.list<i32>):
  //   %v = vm.list.get.i32 %list_0, %c0 : (!vm.list<i32>, i32) -> i32
  //   vm.check.eq %v, %c1 : i32
  //   vm.return
  // ^bb2:
    vm.return
  }
>>>>>>> Wrap lists in iree_vm_ref_t
}
