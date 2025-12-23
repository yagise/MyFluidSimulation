// main_hybrid.cpp
//
// 従来法 + HOME 法の Hybrid(B0) を回すための headless 実行ファイル。
//
// Hybrid(B0) の定義（このリポジトリにおける意味）:
// - 壁近傍(dist_to_solid <= d0) : 従来法(Legacy)の衝突
// - それ以外 : HOME 法の衝突
//
// 注意:
// - これは新しい物理モデルではなく同一のストリーミングと
// セルごとの衝突演算の切り替えを行う比較用実装です。
//

#include "apps/sim_runner.hpp"

#include "lbm/lbm3d_hybrid.hpp"

// summary: 実行エントリポイントとしてシミュレーションを開始する
// param argc: 入力パラメータ
// param argv: 入力パラメータ
// return: 終了コード
int main(int argc, char** argv) {
    DomainConfig dom;
    ObstacleConfig obs;
    RunConfig run;
    HybridConfig hyb;

    // Hybrid は d0 が必要なので HybridConfig を渡す
    parse_common_args(argc, argv, dom, obs, run, &hyb);

    return run_hybrid_b0<LBM3D_Hybrid>(dom, obs, run, hyb);
}
