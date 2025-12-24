// main_legacy.cpp（従来法エントリ）
//
// 従来法（Legacy/BGK）単体を回すためのヘッドレス実行ファイル。
//
// - 従来法 / HOME / Hybrid(B0) をそれぞれ別ファイルの main として分離する
// - 手続き生成の障害物は扱わない（必要なら --stl を明示）
// - 回転体・移動壁などのデバッグ機能は入れない
//

#include "apps/sim_runner.hpp"

#include "lbm/lbm3d_legacy.hpp"
// 実行エントリポイントとしてシミュレーションを開始する
int main(int argc, char** argv) {
    DomainConfig dom;
    ObstacleConfig obs;
    RunConfig run;

    // 従来法では HybridConfig は不要なので nullptr を渡す
    parse_common_args(argc, argv, dom, obs, run, nullptr);

    return run_single_solver<LBM3D_Legacy>(dom, obs, run, "legacy");
}

