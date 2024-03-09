#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler {

namespace {

struct TESTREWRITEPass final : public TESTREWRITEBase<TESTREWRITEPass> {

  void runOnOperation() override {
    auto executableOp = getOperation();

    IRRewriter rewriter(&getContext());
    rewriter.setInsertionPoint(executableOp);
    for (auto variantOp :
         executableOp.getOps<IREE::HAL::ExecutableVariantOp>()) {
      auto moduleOp = variantOp.getInnerModule();
      for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
        IRMapping mapper;
        auto newFuncOp = rewriter.clone(*funcOp.getOperation(), mapper);
        newFuncOp->setAttr("hal.executable.target", variantOp.getTarget());
        if (auto exportOp = getEntryPoint(funcOp)) {
          auto translationInfo = exportOp.value()->getAttr("translation_info");
          if (translationInfo) {
            newFuncOp->setAttr("translation_info", translationInfo);
          }
        }
      }
    }
    rewriter.eraseOp(executableOp);
  }
};

} // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableOp>>
createTESTREWRITEPass() {
  return std::make_unique<TESTREWRITEPass>();
}

} // namespace mlir::iree_compiler