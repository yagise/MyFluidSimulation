// main_home.cpp
//
// HOME 法単体を回すための headless 実行ファイル。
//
// 論文用の整理方針:
// - HOME 法は論文の記述に忠実な最小構成で実行できるようにする
// - 回転体・移動壁・デバッグ用自動モデル投入などは行わない
//

#include "apps/sim_runner.hpp"

#include "lbm/lbm3d_home.hpp"

// summary: 実行エントリポイントとしてシミュレーションを開始する
// param argc: 入力パラメータ
// param argv: 入力パラメータ
// return: 終了コード
int main(int argc, char** argv) {
    DomainConfig dom;
    ObstacleConfig obs;
    RunConfig run;

    parse_common_args(argc, argv, dom, obs, run, nullptr);
    return run_single_solver<LBM3D_Home>(dom, obs, run, "home");
}
