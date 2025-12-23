// main_legacy.cpp
//
// 従来法（Legacy/BGK）単体を回すための headless 実行ファイル。
//
// 論文用の整理方針:
// - 従来法 / HOME / Hybrid(B0) をそれぞれ別ファイルの main として分離する
// - fan/teardrop の自動投入は行わない（必要なら --fan/--teardrop/--stl を明示）
// - 回転体・移動壁などのデバッグ機能は入れない
//

#include "apps/sim_runner.hpp"

#include "lbm/lbm3d_legacy.hpp"

// summary: 実行エントリポイントとしてシミュレーションを開始する
// param argc: 入力パラメータ
// param argv: 入力パラメータ
// return: 終了コード
int main(int argc, char** argv) {
    DomainConfig dom;
    ObstacleConfig obs;
    RunConfig run;

    // 従来法では HybridConfig は不要なので nullptr を渡す
    parse_common_args(argc, argv, dom, obs, run, nullptr);

    return run_single_solver<LBM3D_Legacy>(dom, obs, run, "legacy");
}
