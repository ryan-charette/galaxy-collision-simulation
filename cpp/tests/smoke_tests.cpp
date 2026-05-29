#include <iostream>

int run_math_direct_tests();
int run_tree_fmm_accuracy_tests();
int run_cuda_fallback_tests();
int run_config_snapshot_tests();

int main() {
    int failures = 0;
    failures += run_math_direct_tests();
    failures += run_tree_fmm_accuracy_tests();
    failures += run_cuda_fallback_tests();
    failures += run_config_snapshot_tests();

    if (failures != 0) {
        std::cerr << failures << " smoke test checks failed\n";
        return 1;
    }

    std::cout << "smoke_tests passed\n";
    return 0;
}
