// 修正ポイント: マクロ定義で衝突を先に潰す
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define GLFW_INCLUDE_NONE  // GLFW が OpenGL ヘッダを勝手に入れないように

#include <algorithm>  // std::max
#include <filesystem>
#include <limits>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

#include <glad/glad.h>        // GLAD は GLFW より先
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "render/window.hpp"
#include "render/Vis3D.hpp"
#include "render/MeshRenderer.hpp"
#include "geom/stl_loader.hpp"
#include "geom/voxelize.hpp"
#include "lbm/lbm3d_legacy.hpp"
#include "lbm/lbm3d_home.hpp"
#include "lbm/lbm3d_hybrid.hpp"

extern "C" void launch_shift(const float* a, float bias, float* out, int N);
extern "C" void launch_speed(const float* u, const float* v, const float* w, float* out, int N);
extern "C" void launch_diff(const float* a, const float* b, float* out, int N);
extern "C" void write_scalar_vtk(const char* filename, const float* d_field, int Nx, int Ny, int Nz);
extern "C" void write_vector_vtk(const char* filename,
                                 const float* d_fx,
                                 const float* d_fy,
                                 const float* d_fz,
                                 int Nx, int Ny, int Nz);
extern "C" void reinit_equilibrium_from_macro(const float* d_rho,
                                              const float* d_ux,
                                              const float* d_uy,
                                              const float* d_uz,
                                              float*       d_f,
                                              int          N);
extern "C" void make_rho_gauss(float* d_rho, int Nx,int Ny,int Nz,
                               float cx,float cy,float cz, float sigma, float amp);
extern "C" void make_rho_slab(float* d_rho, int Nx,int Ny,int Nz,
                              int axis, float amp, float center, float width);

struct Camera {
    glm::vec3 pos    = glm::vec3(2.0f, 2.0f, 2.0f);
    glm::vec3 target = glm::vec3(0.5f, 0.5f, 0.5f);
    glm::mat4 P{}, V{};
    // summary: 状態を更新する
    // param w: 入力パラメータ
    // param h: 入力パラメータ
    // return: なし
    void update(int w, int h){
        float aspect = (h != 0) ? (w / float(h)) : 1.0f;
        P = glm::perspective(glm::radians(45.0f), aspect, 0.01f, 10.0f);
        V = glm::lookAt(pos, target, glm::vec3(0,1,0));
    }
};
struct OrbitState {
    float yaw   = 0.0f;   // ラジアン
    float pitch = 0.3f;   // ラジアン（上向き正）
    float radius = 2.5f;  // 目標点からの距離
    glm::vec3 target = glm::vec3(0.5f, 0.5f, 0.5f);

    bool rotating = false;
    bool dollying = false;
    double lastX = 0.0, lastY = 0.0;
};

struct Tracer {
    glm::vec3 pos;
    std::vector<glm::vec3> trail;
};

// summary: rotate_mesh_z の処理を行う
// param src: 入力パラメータ
// param angle: 入力パラメータ
// return: 戻り値
static TriangleMesh rotate_mesh_z(const TriangleMesh& src, float angle){
    TriangleMesh dst;
    dst.indices = src.indices;
    dst.bbmin = glm::vec3( std::numeric_limits<float>::max() );
    dst.bbmax = glm::vec3( -std::numeric_limits<float>::max() );
    float c = std::cos(angle);
    float s = std::sin(angle);
    dst.positions.reserve(src.positions.size());
    for(const auto& p : src.positions){
        glm::vec3 r(c*p.x - s*p.y, s*p.x + c*p.y, p.z);
        dst.positions.push_back(r);
        dst.bbmin = glm::vec3(std::min(dst.bbmin.x, r.x), std::min(dst.bbmin.y, r.y), std::min(dst.bbmin.z, r.z));
        dst.bbmax = glm::vec3(std::max(dst.bbmax.x, r.x), std::max(dst.bbmax.y, r.y), std::max(dst.bbmax.z, r.z));
    }
    return dst;
}
// summary: translate_mesh の処理を行う
// param src: 入力パラメータ
// param delta: 入力パラメータ
// return: 戻り値
static TriangleMesh translate_mesh(const TriangleMesh& src, const glm::vec3& delta){
    TriangleMesh dst;
    dst.indices = src.indices;
    dst.bbmin = glm::vec3( std::numeric_limits<float>::max() );
    dst.bbmax = glm::vec3( -std::numeric_limits<float>::max() );
    dst.positions.reserve(src.positions.size());
    for(const auto& p : src.positions){
        glm::vec3 t = p + delta;
        dst.positions.push_back(t);
        dst.bbmin = glm::vec3(std::min(dst.bbmin.x, t.x), std::min(dst.bbmin.y, t.y), std::min(dst.bbmin.z, t.z));
        dst.bbmax = glm::vec3(std::max(dst.bbmax.x, t.x), std::max(dst.bbmax.y, t.y), std::max(dst.bbmax.z, t.z));
    }
    return dst;
}

struct TracerRenderer {
    GLuint prog=0, vao=0, vbo=0;
    GLint locMVP=-1, locColor=-1;
    int maxVerts=0;

    // summary: compileShader の処理を行う
    // param type: 入力パラメータ
    // param src: 入力パラメータ
    // return: 戻り値
    static GLuint compileShader(GLenum type, const char* src){
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok=0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if(!ok){
            char buf[512]; glGetShaderInfoLog(s, 512, nullptr, buf);
            std::fprintf(stderr, "Tracer shader compile error: %s\n", buf);
        }
        return s;
    }
    // summary: 初期化処理を行う
    // param maxPoints: 入力パラメータ
    // return: なし
    void init(int maxPoints){
        maxVerts = maxPoints;
        const char* vsrc =
            "#version 330 core\n"
            "layout(location=0) in vec3 aPos;\n"
            "uniform mat4 uMVP;\n"
            // summary: vec4 の処理を行う
            // param: なし
            // return: 戻り値
            "void main(){ gl_Position = uMVP * vec4(aPos,1.0); }\n";
        const char* fsrc =
            "#version 330 core\n"
            "uniform vec3 uColor;\n"
            "out vec4 fragColor;\n"
            // summary: vec4 の処理を行う
            // param: なし
            // return: 戻り値
            "void main(){ fragColor = vec4(uColor,1.0); }\n";
        GLuint vs = compileShader(GL_VERTEX_SHADER, vsrc);
        GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsrc);
        prog = glCreateProgram();
        glAttachShader(prog, vs);
        glAttachShader(prog, fs);
        glLinkProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
        locMVP   = glGetUniformLocation(prog, "uMVP");
        locColor = glGetUniformLocation(prog, "uColor");

        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float)*3*maxVerts, nullptr, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*3, (void*)0);
        glBindVertexArray(0);
    }
    // summary: 状態を更新する
    // param t: 入力パラメータ
    // return: なし
    void updateBuffer(const Tracer& t){
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        int cnt = (int)std::min<size_t>(t.trail.size(), maxVerts);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*3*cnt, t.trail.data());
    }
    // summary: 描画処理を行う
    // param t: 入力パラメータ
    // param mvp: 入力パラメータ
    // param color: 入力パラメータ
    // return: なし
    void draw(const Tracer& t, const glm::mat4& mvp, const glm::vec3& color){
        if(t.trail.empty()) return;
        int cnt = (int)std::min<size_t>(t.trail.size(), maxVerts);
        updateBuffer(t);
        glUseProgram(prog);
        glUniformMatrix4fv(locMVP, 1, GL_FALSE, glm::value_ptr(mvp));
        glUniform3fv(locColor, 1, glm::value_ptr(color));
        glBindVertexArray(vao);
        glDisable(GL_DEPTH_TEST);
        glLineWidth(2.5f);
        glDrawArrays(GL_LINE_STRIP, 0, cnt);
        glPointSize(8.0f);
        glDrawArrays(GL_POINTS, cnt-1, 1); // 末端だけポイントで強調
        glEnable(GL_DEPTH_TEST);
        glBindVertexArray(0);
    }
};

enum class Display    { Pressure, Speed, Error };
enum class SolverView { Legacy, Home, Hybrid, Compare };

// summary: setTitle の処理を行う
// param win: 入力パラメータ
// param s: 入力パラメータ
// return: なし
static void setTitle(const Window& win, SolverView s){
    const char* m = (s==SolverView::Legacy) ? "Legacy(BGK)"
                 : (s==SolverView::Home   ? "HOME(Moment)"
                 : (s==SolverView::Hybrid ? "Hybrid(B0)"
                 : "Compare"));
    std::string t = std::string("LBM Compare - ") + m +
        "  [1:rho  2:|u|  3:error  G:legacy  H:home  B:hybrid  C:compare  O:obstacle]";
    glfwSetWindowTitle(win.handle(), t.c_str());
}

// summary: 実行エントリポイントとしてシミュレーションを開始する
// param argc: 入力パラメータ
// param argv: 入力パラメータ
// return: 終了コード
int main(int argc, char** argv){
    // シミュレーション領域を広げて障害物との間に余白を確保（元: 160 x 120 x 96）
    Domain d; d.Nx=200; d.Ny=150; d.Nz=120; d.tau=0.58f; d.forceX=3e-6f;
    const int numCells = d.Nx * d.Ny * d.Nz;

    TriangleMesh mesh;
    // 論文用整理:
    // - デバッグ用途の障害物を勝手に入れる挙動は廃止。
    // - 明示的に --stl / --fan / --teardrop が指定されたときだけ障害物を置く。
    // - 何も指定が無い場合は障害物なしで走る。
    bool haveObstacle = false;
    std::string stlPath;

std::string initMode = "none";  // none / gauss / slab
float amp = 1e-3f, sigma = 0.08f;
glm::vec3 cen(0.6f,0.5f,0.5f);
int axis = 0; float center=0.5f, width=0.2f;
std::string vtkOutDir;
int vtkInterval = 0;
enum class VtkSolver { Legacy, Home };
VtkSolver vtkSolver = VtkSolver::Home;
int amrFactor = 1;
float amrThreshold = 0.5f;
int wallMetricsEvery = 0; // steps; 0 disables near-wall diagnostics

// --- Hybrid(B0) parameters ---
int hybridBand = 2;          // B0: wall-distance band thickness (d0). default=2
int hybridSweepMax = -1;     // if >=0, run headless sweep for d0=0..hybridSweepMax and exit
int hybridSweepSteps = 500;  // steps per d0 in sweep mode




    // 既定は従来法。--method / --solver のどちらでも指定可
    SolverView solverView = SolverView::Legacy;

    // 引数処理（衝突しやすい 'i' を避ける）
    for(int argi = 1; argi < argc; ++argi){
        std::string a = argv[argi];
        if(a == "--stl" && argi+1 < argc){
            stlPath = argv[++argi];
            haveObstacle = true;
        }
        else if(a == "--fan"){
            // 手続き生成モデルは明示されたときだけ使用する。
            mesh = make_fan();
            haveObstacle = true;
        }
        else if(a == "--teardrop"){
            mesh = make_teardrop();
            haveObstacle = true;
        }
        else if (a == "--force" && argi+1 < argc) { d.forceX = std::stof(argv[++argi]); }
        else if ((a == "--forcey" || a=="--force-y") && argi+1 < argc) { d.forceY = std::stof(argv[++argi]); }
        else if ((a == "--forcez" || a=="--force-z") && argi+1 < argc) { d.forceZ = std::stof(argv[++argi]); }
        // 論文用整理:
        // - 回転体・落下など、移動壁に由来する追加機能は削除。
        // （論文の比較対象に不要 / 条件を複雑にするため）
        else if((a == "--method" || a=="--solver") && argi+1 < argc){
            std::string s = argv[++argi];
            if(s=="legacy") solverView = SolverView::Legacy;
            else if(s=="home" || s=="moments") solverView = SolverView::Home;
            else if(s=="hybrid" || s=="b0") solverView = SolverView::Hybrid;
            else if(s=="compare") solverView = SolverView::Compare;
        }else if (a == "--init" && argi+1 < argc) { 
    initMode = argv[++argi]; // "gauss" or "slab" or "none"
}
else if (a == "--amp" && argi+1 < argc) { 
    amp = std::stof(argv[++argi]);     // 例: 1e-3
}
else if (a == "--sigma" && argi+1 < argc) {
    sigma = std::stof(argv[++argi]);   // 例: 0.08
}
else if (a == "--center" && argi+3 < argc) {
    // gauss 用の中心 (x y z)
    cen.x = std::stof(argv[++argi]);
    cen.y = std::stof(argv[++argi]);
    cen.z = std::stof(argv[++argi]);
}
else if (a == "--axis" && argi+1 < argc) {
    // slab 用の軸: x|y|z
    char ax = argv[++argi][0];
    axis = (ax=='y') ? 1 : (ax=='z') ? 2 : 0;
}
else if (a == "--width" && argi+1 < argc) { 
    // slab 用の幅
    width = std::stof(argv[++argi]);
}
else if (a == "--vtk-dir" && argi+1 < argc) {
    vtkOutDir = argv[++argi];
}
else if ((a == "--vtk-interval" || a=="--vtk-every") && argi+1 < argc) {
    vtkInterval = std::stoi(argv[++argi]);
}
else if (a == "--vtk-solver" && argi+1 < argc) {
    std::string s = argv[++argi];
    if(s=="legacy") vtkSolver = VtkSolver::Legacy;
    else if(s=="home" || s=="moment") vtkSolver = VtkSolver::Home;
}
else if (a == "--amr"){
    amrFactor = std::max(amrFactor, 2);
}
else if ((a == "--amr-factor" || a=="--amr-refine") && argi+1 < argc) {
    amrFactor = std::stoi(argv[++argi]);
}
else if (a == "--amr-threshold" && argi+1 < argc) {
    amrThreshold = std::stof(argv[++argi]);
}
else if ((a == "--wall-metrics-every" || a=="--wall-metrics") && argi+1 < argc) {
    wallMetricsEvery = std::stoi(argv[++argi]);
}
else if ((a == "--hybrid-band" || a=="--hybrid-d0") && argi+1 < argc) {
    hybridBand = std::max(0, std::stoi(argv[++argi]));
}
else if ((a == "--hybrid-sweep-max" || a=="--hybrid-sweep") && argi+1 < argc) {
    hybridSweepMax = std::max(0, std::stoi(argv[++argi]));
}
else if ((a == "--hybrid-sweep-steps" || a=="--sweep-steps") && argi+1 < argc) {
    hybridSweepSteps = std::max(0, std::stoi(argv[++argi]));
}

    }
    if(amrFactor < 1) amrFactor = 1;
    amrThreshold = std::clamp(amrThreshold, 0.0f, 1.0f);
    if(wallMetricsEvery < 0) wallMetricsEvery = 0;
std::printf("[args] init=%s amp=%.3g sigma=%.3g center=(%.3f,%.3f,%.3f) axis=%d width=%.3f forceX=%.3g\n",
            initMode.c_str(), amp, sigma, cen.x, cen.y, cen.z, axis, width, d.forceX);
std::printf("[args] force=(%.3g, %.3g, %.3g)\n", d.forceX, d.forceY, d.forceZ);
std::printf("[amr ] refine=%d threshold=%.2f\n", amrFactor, amrThreshold);
if(wallMetricsEvery>0){
    std::printf("[wall] metrics every %d steps (near-wall |u|)\n", wallMetricsEvery);
}

    std::filesystem::path vtkDirPath;
    bool vtkEnabled = false;
    if(!vtkOutDir.empty()){
        vtkDirPath = std::filesystem::path(vtkOutDir);
        if(vtkInterval <= 0){
            std::fprintf(stderr, "[vtk ] ignored because interval <= 0\n");
        }else{
            std::error_code ec;
            std::filesystem::create_directories(vtkDirPath, ec);
            if(ec){
                std::fprintf(stderr, "[vtk ] failed to create dir: %s (%s)\n", vtkOutDir.c_str(), ec.message().c_str());
            }else{
                vtkEnabled = true;
                const char* solverName = (vtkSolver==VtkSolver::Legacy)? "legacy" : "home";
                std::printf("[vtk ] output -> %s  every %d steps  solver=%s\n",
                            vtkDirPath.string().c_str(), vtkInterval, solverName);
            }
        }
    }

    // 論文用整理: デフォルトで teardrop を入れる挙動は削除。
    // ここで STL を読み込めなかった場合は実験として意味が無いので終了する。
    if(!stlPath.empty()){
        TriangleMesh m2;
        if(load_stl(stlPath, m2)){
            mesh = std::move(m2);
            haveObstacle = true;
        }else{
            std::fprintf(stderr, "Failed to load STL: %s\n", stlPath.c_str());
            return 1;
        }
    }

    // 障害物のボクセル化（[0,1]^3 空間）
    // ドメイン内で障害物が占める比率を小さくして周囲の余白を確保
    VoxelParams vp; vp.Nx=d.Nx; vp.Ny=d.Ny; vp.Nz=d.Nz;
    vp.uniformScale = 0.5f; vp.translate=glm::vec3(0.5f, 0.5f, 0.5f);
    vp.refine = amrFactor; vp.coverageThreshold = amrThreshold;
    TriangleMesh meshCurrent;
    std::vector<unsigned char> solidMask((size_t)numCells, 0);
    if(haveObstacle){
        meshCurrent = mesh;
        solidMask = voxelize_mesh_to_mask(meshCurrent, vp);
    }
    std::vector<int> nearWallFluid;
    auto recomputeNearWall = [&](const std::vector<unsigned char>& mask){
        if(wallMetricsEvery <= 0 && hybridSweepMax < 0) return;
        nearWallFluid.clear();
        nearWallFluid.reserve(mask.size()/10);
        const size_t strideY = d.Nx;
        const size_t strideZ = (size_t)d.Nx * (size_t)d.Ny;
        auto idx = [&](int x,int y,int z)->size_t{
            return (size_t)x + strideY * ((size_t)y + (size_t)d.Ny * (size_t)z);
        };
        for(int z=0; z<d.Nz; ++z){
            for(int y=0; y<d.Ny; ++y){
                for(int x=0; x<d.Nx; ++x){
                    size_t i = idx(x,y,z);
                    if(mask[i]) continue;
                    bool near = false;
                    if(x>0 && mask[i-1]) near=true;
                    else if(x<d.Nx-1 && mask[i+1]) near=true;
                    else if(y>0 && mask[i - d.Nx]) near=true;
                    else if(y<d.Ny-1 && mask[i + d.Nx]) near=true;
                    else if(z>0 && mask[i - strideZ]) near=true;
                    else if(z<d.Nz-1 && mask[i + strideZ]) near=true;
                    if(near) nearWallFluid.push_back((int)i);
                }
            }
        }
        std::printf("[wall] near-wall fluid cells: %zu\n", nearWallFluid.size());
    };
    recomputeNearWall(solidMask);


//
// Hybrid(B0) preprocessing
//
// We need an integer "distance to solid" field (6-neighborhood) so that we can
// mark the near-wall band as LEGACY cells:
//
// legacy if (distance_to_solid <= d0)
// HOME otherwise
//
// NOTE: This distance is computed with non-periodic boundaries (same as the
// existing nearWallFluid diagnostic). If your obstacle touches the domain
// boundary and you rely on periodicity, consider changing this to wraparound.
const int maxBandForDist = std::max(hybridBand, hybridSweepMax);
std::vector<int> distToSolid(numCells, -1);
if(maxBandForDist > 0){
    // Multi-source BFS from all solid cells (distance 0).
    std::vector<int> q;
    q.reserve(numCells/4);
    for(int i=0;i<numCells;++i){
        if(solidMask[i]){
            distToSolid[i] = 0;
            q.push_back(i);
        }
    }
    size_t head = 0;
    const int Nx = d.Nx;
    const int Ny = d.Ny;
    const int Nz = d.Nz;
    const int strideY = Nx;
    const int strideZ = Nx*Ny;

    auto pushIfUnvisited = [&](int nid, int nd){
        if(distToSolid[nid] == -1){
            distToSolid[nid] = nd;
            q.push_back(nid);
        }
    };

    while(head < q.size()){
        const int id   = q[head++];
        const int dcur = distToSolid[id];
        if(dcur >= maxBandForDist) continue; // we only need distances up to maxBandForDist

        const int z = id / strideZ;
        const int rem = id - z*strideZ;
        const int y = rem / strideY;
        const int x = rem - y*strideY;
        const int nd = dcur + 1;

        if(x > 0)      pushIfUnvisited(id - 1,        nd);
        if(x < Nx - 1) pushIfUnvisited(id + 1,        nd);
        if(y > 0)      pushIfUnvisited(id - strideY,  nd);
        if(y < Ny - 1) pushIfUnvisited(id + strideY,  nd);
        if(z > 0)      pushIfUnvisited(id - strideZ,  nd);
        if(z < Nz - 1) pushIfUnvisited(id + strideZ,  nd);
    }
}

// Build mapping arrays for a given band thickness d0.
// Output:
// isLegacyOut[i] = 1 if fluid cell i is in the legacy band
// slotOut[i] = compact index in [0, nLegacyCells)
auto buildLegacyBand = [&](int d0,
                           std::vector<unsigned char>& isLegacyOut,
                           std::vector<int>&           slotOut)->int
{
    isLegacyOut.assign(numCells, 0);
    slotOut.assign(numCells, -1);

    int count = 0;
    for(int i=0;i<numCells;++i){
        if(solidMask[i]) continue;
        const int di = distToSolid[i];
        if(di >= 0 && di <= d0){
            isLegacyOut[i] = 1;
            slotOut[i] = count++;
        }
    }
    return count;
};

// Build the mapping for the "interactive" hybrid solver instance (single d0).
std::vector<unsigned char> hybridIsLegacy;
std::vector<int>           hybridLegacySlot;
const int nLegacyCells = buildLegacyBand(hybridBand, hybridIsLegacy, hybridLegacySlot);
std::printf("[hybrid] B0 band d0=%d -> legacy cells=%d (fluid only)\n", hybridBand, nLegacyCells);


// 3系統の LBM
LBM3D_Legacy legacy; legacy.init(d); legacy.setSolidMask(solidMask.data());
LBM3D_Home   home;   home.init(d);   home.setSolidMask(solidMask.data());
LBM3D_Hybrid hybrid; hybrid.init(d, nLegacyCells); hybrid.setSolidMask(solidMask.data());
hybrid.setLegacyMapping(hybridIsLegacy.data(), hybridLegacySlot.data());

    int N = d.Nx*d.Ny*d.Nz;

// d_rhoInit を作成
float* d_rhoInit=nullptr; cudaMalloc(&d_rhoInit, sizeof(float)*N);

// ρ=1 の場に初期バンプを作る
if(initMode=="gauss"){
    make_rho_gauss(d_rhoInit, d.Nx,d.Ny,d.Nz, cen.x,cen.y,cen.z, sigma, amp);
}else if(initMode=="slab"){
    make_rho_slab(d_rhoInit, d.Nx,d.Ny,d.Nz, axis, amp, center, width);
}else{ // none
    // ρ=1 で開始
    cudaMemset(d_rhoInit, 0, sizeof(float)*N);
    // 小さなカーネルで+1しても良いが、ここでは reinit 側で rho=0 を許容しないので、
    // 簡易には legacy.d_rho() を memcpy して使ってもOK
    cudaMemcpy(d_rhoInit, legacy.d_rho(), sizeof(float)*N, cudaMemcpyDeviceToDevice);
}


// --- 平衡に “再構成” ---
// 論文用整理:
// 初期条件の作り方を全ソルバで統一する。
// - Legacy : (rho,u) から平衡分布 f_i を構成し、fnext も整合
// - HOME : moments-only なので (rho,u,S=0) を構成し、mnext も整合
legacy.reinitEquilibriumFromMacro(d_rhoInit, nullptr, nullptr, nullptr);
home  .reinitEquilibriumFromMacro(d_rhoInit, nullptr, nullptr, nullptr);
hybrid.reinitEquilibriumFromMacro(d_rhoInit, nullptr, nullptr, nullptr);


//
// Hybrid d0 sweep (headless)
//
// If --hybrid-sweep-max is provided, we skip the GUI and quantify how much
// the band thickness d0 affects the near-wall velocity statistics.
//
// Output is CSV to stdout:
// d0, legacy_cells, mean|u|_legacy, mean|u|_hybrid, mean_abs_diff
//
// This is intentionally simple so you can post-process it with Python/R.
if(hybridSweepMax >= 0){
    //
    // Hybrid B0 study sweep (headless)
    //
    // This mode is intended for *experiments* (no GUI):
    //
    // - Run a LEGACY baseline once (full f_i everywhere)
    // - Run a HOME baseline once (moment-encoded; reconstruct f_i on the fly)
    // - For d0 = 0..hybridSweepMax:
    // run HYBRID(B0) where near-wall band (dist_to_solid <= d0) is LEGACY
    // and the bulk is HOME-style moment cache.
    //
    // We then report near-wall statistics and how close HOME/HYBRID are to LEGACY.
    //
    // Output: CSV to stdout (easy to redirect to file on Windows).
    //
    // NOTE:
    // - "near-wall" here means fluid cells with a 6-neighbor solid cell (dist=1).
    // - "legacy_ratio" is legacy_cells / fluid_cells (not counting solid cells).
    //
    std::printf("[hybrid] B0 sweep enabled: d0=0..%d, steps_per_run=%d\n",
                hybridSweepMax, hybridSweepSteps);

    // Count fluid cells once for ratio reporting.
    int nFluidCells = 0;
    for(int i=0;i<numCells;++i) if(!solidMask[i]) ++nFluidCells;

    // Scratch buffers to compute |u| and mean rho, and copy back.
    float* d_speed = nullptr;
    cudaMalloc(&d_speed, sizeof(float)*numCells);
    std::vector<float> hSpeed(numCells);

    std::vector<float> hRho(numCells);

    auto mean_rho_from_device = [&](const float* d_rho)->double{
        cudaMemcpy(hRho.data(), d_rho, sizeof(float)*numCells, cudaMemcpyDeviceToHost);
        double sum = 0.0;
        for(int i=0;i<numCells;++i){
            if(!solidMask[i]) sum += (double)hRho[i];
        }
        return (nFluidCells > 0) ? (sum / (double)nFluidCells) : 0.0;
    };

    // Helper: time "steps" calls to solver.step(1) using CUDA events.
    auto time_ms_per_step = [&](auto&& stepLambda, int steps)->double{
        if(steps <= 0) return 0.0;
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
        cudaEventRecord(ev0, 0);
        for(int s=0; s<steps; ++s) stepLambda();
        cudaEventRecord(ev1, 0);
        cudaEventSynchronize(ev1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev0, ev1);
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        return (double)ms / (double)steps;
    };

    //
    // 1) LEGACY baseline (run once)
    //
    std::vector<float> baselineNearWall; baselineNearWall.reserve(nearWallFluid.size());
    double meanU_legacy = 0.0;
    double meanRho_legacy = 0.0;
    double msPerStep_legacy = 0.0;

    {
        LBM3D_Legacy leg;
        leg.init(d);
        leg.setSolidMask(solidMask.data());

        // Initial condition: equilibrium reconstructed from rho-field
        leg.reinitEquilibriumFromMacro(d_rhoInit, nullptr, nullptr, nullptr);

        // summary: step の処理を行う
        // param [&](: 入力パラメータ
        // return: 戻り値
        msPerStep_legacy = time_ms_per_step([&](){ leg.step(1); }, hybridSweepSteps);

        // Compute speed field
        launch_speed(leg.d_u(), leg.d_v(), leg.d_w(), d_speed, numCells);
        cudaMemcpy(hSpeed.data(), d_speed, sizeof(float)*numCells, cudaMemcpyDeviceToHost);

        // Extract near-wall speeds (baseline) and compute mean
        double sumL = 0.0;
        baselineNearWall.resize(nearWallFluid.size());
        for(size_t k=0; k<nearWallFluid.size(); ++k){
            const int idx = nearWallFluid[k];
            const float sl = hSpeed[idx];
            baselineNearWall[k] = sl;
            sumL += (double)sl;
        }
        const double cnt = (nearWallFluid.empty()? 1.0 : (double)nearWallFluid.size());
        meanU_legacy = sumL / cnt;

        // Mean rho (mass conservation sanity check)
        meanRho_legacy = mean_rho_from_device(leg.d_rho());
    } // leg destructor frees GPU memory

    //
    // 2) HOME baseline (run once)
    //
    double meanU_home = 0.0;
    double meanAbsDiff_home = 0.0;
    double meanRho_home = 0.0;
    double msPerStep_home = 0.0;

    {
        LBM3D_Home ho;
        ho.init(d);
        ho.setSolidMask(solidMask.data());

        // HOME も同じ API で初期条件を作る（moments-only）
        ho.reinitEquilibriumFromMacro(d_rhoInit, nullptr, nullptr, nullptr);

        // summary: step の処理を行う
        // param [&](: 入力パラメータ
        // return: 戻り値
        msPerStep_home = time_ms_per_step([&](){ ho.step(1); }, hybridSweepSteps);

        launch_speed(ho.d_u(), ho.d_v(), ho.d_w(), d_speed, numCells);
        cudaMemcpy(hSpeed.data(), d_speed, sizeof(float)*numCells, cudaMemcpyDeviceToHost);

        double sumH = 0.0, sumAbs = 0.0;
        for(size_t k=0; k<nearWallFluid.size(); ++k){
            const int idx = nearWallFluid[k];
            const float sh = hSpeed[idx];
            sumH   += (double)sh;
            sumAbs += (double)std::abs(sh - baselineNearWall[k]);
        }
        const double cnt = (nearWallFluid.empty()? 1.0 : (double)nearWallFluid.size());
        meanU_home = sumH / cnt;
        meanAbsDiff_home = sumAbs / cnt;

        meanRho_home = mean_rho_from_device(ho.d_rho());
    } // ho destructor frees GPU memory

    //
    // 3) HYBRID(B0) sweep for d0=0..max
    //
    // CSV header:
    std::printf("d0,legacy_cells,legacy_ratio,"
                "mean_u_legacy,mean_u_home,mean_u_hybrid,"
                "mean_absdiff_home,mean_absdiff_hybrid,"
                "mean_rho_legacy,mean_rho_home,mean_rho_hybrid,"
                "absdiff_rho_home,absdiff_rho_hybrid,"
                "ms_per_step_legacy,ms_per_step_home,ms_per_step_hybrid,"
                // --- extra columns to compare *wall AMR* settings across runs ---
                // These make it easier to merge/plot CSVs without relying on file names.
                "amr_factor,amr_threshold,fluid_cells,near_wall_cells\n");

    for(int d0=0; d0<=hybridSweepMax; ++d0){
        std::vector<unsigned char> isLegacy;
        std::vector<int>           slot;
        const int nLeg = buildLegacyBand(d0, isLegacy, slot);
        const double legacyRatio = (nFluidCells > 0) ? ((double)nLeg / (double)nFluidCells) : 0.0;

        double meanU_hyb = 0.0;
        double meanAbsDiff_hyb = 0.0;
        double meanRho_hyb = 0.0;
        double msPerStep_hyb = 0.0;

        {
            LBM3D_Hybrid hyb;
            hyb.init(d, nLeg);
            hyb.setSolidMask(solidMask.data());
            hyb.setLegacyMapping(isLegacy.data(), slot.data());

            hyb.reinitEquilibriumFromMacro(d_rhoInit, nullptr, nullptr, nullptr);

            // summary: step の処理を行う
            // param [&](: 入力パラメータ
            // return: 戻り値
            msPerStep_hyb = time_ms_per_step([&](){ hyb.step(1); }, hybridSweepSteps);

            launch_speed(hyb.d_u(), hyb.d_v(), hyb.d_w(), d_speed, numCells);
            cudaMemcpy(hSpeed.data(), d_speed, sizeof(float)*numCells, cudaMemcpyDeviceToHost);

            double sum = 0.0, sumAbs = 0.0;
            for(size_t k=0; k<nearWallFluid.size(); ++k){
                const int idx = nearWallFluid[k];
                const float sh = hSpeed[idx];
                sum    += (double)sh;
                sumAbs += (double)std::abs(sh - baselineNearWall[k]);
            }
            const double cnt = (nearWallFluid.empty()? 1.0 : (double)nearWallFluid.size());
            meanU_hyb = sum / cnt;
            meanAbsDiff_hyb = sumAbs / cnt;

            meanRho_hyb = mean_rho_from_device(hyb.d_rho());
        } // hyb destructor frees GPU memory

        const double absDiffRho_home = std::abs(meanRho_home - meanRho_legacy);
        const double absDiffRho_hyb  = std::abs(meanRho_hyb  - meanRho_legacy);

        std::printf("%d,%d,%.6f,"
                    "%.9e,%.9e,%.9e,"
                    "%.9e,%.9e,"
                    "%.9e,%.9e,%.9e,"
                    "%.9e,%.9e,"
                    "%.6f,%.6f,%.6f,"
                    "%d,%.6f,%d,%zu\n",
                    d0, nLeg, legacyRatio,
                    meanU_legacy, meanU_home, meanU_hyb,
                    meanAbsDiff_home, meanAbsDiff_hyb,
                    meanRho_legacy, meanRho_home, meanRho_hyb,
                    absDiffRho_home, absDiffRho_hyb,
                    msPerStep_legacy, msPerStep_home, msPerStep_hyb,
                    amrFactor, (double)amrThreshold, nFluidCells, nearWallFluid.size());
    }

    cudaFree(d_speed);
    cudaFree(d_rhoInit);
    return 0;
}



cudaFree(d_rhoInit);


    // 表示関連
    Window win(1280, 720);
Vis3D vis(d.Nx, d.Ny, d.Nz);
    MeshRenderer meshR;
    auto uploadMeshToRenderer = [&](const TriangleMesh& m, const VoxelParams& params){
        glm::vec3 bbsize = m.bbmax - m.bbmin;
        float maxDim = std::max(bbsize.x, std::max(bbsize.y, bbsize.z));
        float s = (params.uniformScale) / (maxDim > 0 ? maxDim : 1.0f);
        glm::vec3 centera = 0.5f*(m.bbmin + m.bbmax);
        glm::vec3 trans  = params.translate - s*centera;
        std::vector<glm::vec3> pos(m.positions.size());
        for(size_t k=0;k<pos.size();++k) pos[k] = s*m.positions[k] + trans;
        meshR.upload(pos, m.indices);
    };
    uploadMeshToRenderer(meshCurrent, vp);
    meshR.setAlpha(1.0f);  // ← 0.25f から 1.0f に


    // 一時バッファ
float *d_speedLegacy=nullptr, *d_speedHome=nullptr, *d_diffRho=nullptr, *d_tmp=nullptr;
cudaMalloc(&d_speedLegacy, sizeof(float)*numCells);
cudaMalloc(&d_speedHome,   sizeof(float)*numCells);
cudaMalloc(&d_diffRho,     sizeof(float)*numCells);
cudaMalloc(&d_tmp,         sizeof(float)*numCells);   // 追加
std::vector<float> hSpeedLegacyHost, hSpeedHomeHost;

const int stepsPerFrame = 2;  // Keep HOME and Legacy stepping together every frame
auto stepBothSolvers = [&](){
    legacy.step(stepsPerFrame);
    home.step(stepsPerFrame);
    hybrid.step(stepsPerFrame);
};
    auto vtkPath = [&](const std::string& stem, int step)->std::string{
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%s_%06d.vtk", stem.c_str(), step);
        return (vtkDirPath / buf).string();
    };
    auto dumpVtk = [&](int step){
        if(!vtkEnabled) return;
        const bool useLegacy = (vtkSolver == VtkSolver::Legacy);
        const float* rhoSrc = useLegacy ? legacy.d_rho() : home.d_rho();
        const float* uxSrc  = useLegacy ? legacy.d_u()   : home.d_u();
        const float* uySrc  = useLegacy ? legacy.d_v()   : home.d_v();
        const float* uzSrc  = useLegacy ? legacy.d_w()   : home.d_w();
        float* speedBuf     = useLegacy ? d_speedLegacy  : d_speedHome;
        const char* solverName = useLegacy ? "legacy" : "home";

        launch_speed(uxSrc, uySrc, uzSrc, speedBuf, numCells);
        std::string rhoFile  = vtkPath(std::string("rho_")   + solverName, step);
        std::string spdFile  = vtkPath(std::string("speed_") + solverName, step);
        std::string velFile  = vtkPath(std::string("vel_")   + solverName, step);
        write_scalar_vtk(rhoFile.c_str(),  rhoSrc,   d.Nx, d.Ny, d.Nz);
        write_scalar_vtk(spdFile.c_str(),  speedBuf, d.Nx, d.Ny, d.Nz);
        write_vector_vtk(velFile.c_str(),  uxSrc, uySrc, uzSrc, d.Nx, d.Ny, d.Nz);
        std::printf("[vtk ] step %d -> %s, %s, %s\n", step, rhoFile.c_str(), spdFile.c_str(), velFile.c_str());
    };
    int simStep = 0;
    int nextVtkDump = vtkInterval;
    if(vtkEnabled){
        dumpVtk(0); // initial state
    }
    // 論文用整理: 障害物を動かす機能（回転/落下）を削除したため、角度状態は不要。

    // トレーサ（Legacy / HOME 両方の速度場に1粒子ずつ乗せる）
    const glm::vec3 tracerInit(0.55f, 0.5f, 0.5f);  // 流速が出やすい管中心付近に寄せる
    const int tracerTrailMax = 120;               // 短めの履歴
    Tracer tracerLegacy{tracerInit, {tracerInit}};
    Tracer tracerHome  {tracerInit, {tracerInit}};
    TracerRenderer tracerR; tracerR.init(tracerTrailMax);
    const float tracerVelScale = 200.0f; // 可視化用に速度を強調（大きめ）
    bool tracerProjectToX = false;   // 斜め成分を無視して +X 成分だけにする
    auto sampleVel = [&](const float* du, const float* dv, const float* dw, const glm::vec3& p)->glm::vec3{
        // 境界セルを避けるため 1..N-2 にクリップ
        int ix = std::clamp(int(p.x * d.Nx), 1, d.Nx-2);
        int iy = std::clamp(int(p.y * d.Ny), 1, d.Ny-2);
        int iz = std::clamp(int(p.z * d.Nz), 1, d.Nz-2);
        size_t idx = (size_t)ix + (size_t)d.Nx * ( (size_t)iy + (size_t)d.Ny * (size_t)iz );
        float ux=0, uy=0, uz=0;
        cudaMemcpy(&ux, du+idx, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&uy, dv+idx, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&uz, dw+idx, sizeof(float), cudaMemcpyDeviceToHost);
        return glm::vec3(ux, uy, uz);
    };
    auto stepTracer = [&](Tracer& t, const float* du, const float* dv, const float* dw){
        // ソルバ計算完了を確実に待つ
        cudaDeviceSynchronize();
        glm::vec3 u = sampleVel(du,dv,dw,t.pos);
        if(tracerProjectToX){ u.y = 0.0f; u.z = 0.0f; }
        // 速度は格子単位/step なので [0,1]^3 正規化に換算しつつ強調
        t.pos += tracerVelScale * glm::vec3(u.x / d.Nx, u.y / d.Ny, u.z / d.Nz) * float(stepsPerFrame);
        t.pos = glm::clamp(t.pos, glm::vec3(0.0f), glm::vec3(1.0f));
        t.trail.push_back(t.pos);
        if((int)t.trail.size() > tracerTrailMax) t.trail.erase(t.trail.begin());
    };

    Camera  cam;
    Display mode = Display::Speed;
    bool showObstacle = true;
    setTitle(win, solverView);
OrbitState orb;
orb.target = cam.target;

// 現在のカメラ位置から yaw/pitch/radius を求めて初期化
{
    glm::vec3 d = cam.pos - orb.target;
    orb.radius = glm::length(d);
    if(orb.radius < 1e-6f) orb.radius = 1.0f;
    // yaw: x-z 平面角度, pitch: 上下角
    orb.yaw   = std::atan2(d.z, d.x);
    orb.pitch = std::asin( glm::clamp(d.y / orb.radius, -1.0f, 1.0f) );
}

// 任意：カーソルの挙動は通常のまま
// glfwSetInputMode(win.handle(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);

// タイトルにヒントを足したい場合（任意）
// setTitle(win, solverView); の実装を編集できるなら末尾に " [LMB:orbit RMB:zoom]" を足してください。
    bool toggleProjPrev = false; // tracerProjectToX toggle key (P)
    while(!win.shouldClose()){
        // 入力（手法と表示の切替）
        if(glfwGetKey(win.handle(), GLFW_KEY_G) == GLFW_PRESS) { solverView = SolverView::Legacy;  setTitle(win, solverView); }
        if(glfwGetKey(win.handle(), GLFW_KEY_H) == GLFW_PRESS) { solverView = SolverView::Home;    setTitle(win, solverView); }
        if(glfwGetKey(win.handle(), GLFW_KEY_B) == GLFW_PRESS) { solverView = SolverView::Hybrid;  setTitle(win, solverView); }
        if(glfwGetKey(win.handle(), GLFW_KEY_C) == GLFW_PRESS) { solverView = SolverView::Compare; setTitle(win, solverView); }

        if(glfwGetKey(win.handle(), GLFW_KEY_1) == GLFW_PRESS) mode = Display::Pressure;
        if(glfwGetKey(win.handle(), GLFW_KEY_2) == GLFW_PRESS) mode = Display::Speed;
        if(glfwGetKey(win.handle(), GLFW_KEY_3) == GLFW_PRESS) mode = Display::Error;
        if(glfwGetKey(win.handle(), GLFW_KEY_O) == GLFW_PRESS) showObstacle = !showObstacle;
        int keyP = glfwGetKey(win.handle(), GLFW_KEY_P);
        if(keyP == GLFW_PRESS && !toggleProjPrev){
            tracerProjectToX = !tracerProjectToX;
            std::printf("[tracer] project-to-X %s\n", tracerProjectToX ? "ON" : "OFF");
        }
        toggleProjPrev = (keyP == GLFW_PRESS);

        // Keep HOME and Legacy in lockstep so view switching never desynchronizes steps
        stepBothSolvers();
        // トレーサ更新（Compare 以外でも毎フレーム両方動かす）
        stepTracer(tracerLegacy, legacy.d_u(), legacy.d_v(), legacy.d_w());
        stepTracer(tracerHome,   home.d_u(),   home.d_v(),   home.d_w());
        simStep += stepsPerFrame;
        if(vtkEnabled && simStep >= nextVtkDump){
            dumpVtk(simStep);
            nextVtkDump += vtkInterval;
        }
        // デバッグ: 一定間隔でトレーサ位置の生の速度を出力
        {
            static int dbg = 0;
            if((dbg++ % 120) == 0){
                glm::vec3 uL = sampleVel(legacy.d_u(), legacy.d_v(), legacy.d_w(), tracerLegacy.pos);
                glm::vec3 uH = sampleVel(home.d_u(),   home.d_v(),   home.d_w(),   tracerHome.pos);
                std::printf("[dbg] pos=(%.3f,%.3f,%.3f)  Legacy u=(%.3e, %.3e, %.3e)  Home u=(%.3e, %.3e, %.3e)\n",
                           tracerLegacy.pos.x, tracerLegacy.pos.y, tracerLegacy.pos.z,
                           uL.x, uL.y, uL.z, uH.x, uH.y, uH.z);
                // 全セルの平均・最大 |v| をざっくり確認
                static std::vector<float> hbuf;
                hbuf.resize(numCells);
                auto dumpFieldStats = [&](const char* name, const float* dfield){
                    cudaMemcpy(hbuf.data(), dfield, sizeof(float)*numCells, cudaMemcpyDeviceToHost);
                    double sum=0.0;
                    float maxabs=0.0f;
                    for(float v : hbuf){ sum += v; maxabs = std::max(maxabs, std::abs(v)); }
                    std::printf("[dbg] %s mean=%.3e  maxabs=%.3e\n", name, sum/numCells, maxabs);
                };
                dumpFieldStats("Legacy ux", legacy.d_u());
                dumpFieldStats("Legacy uy", legacy.d_v());
                dumpFieldStats("Legacy uz", legacy.d_w());
                dumpFieldStats("Home   ux", home.d_u());
                dumpFieldStats("Home   uy", home.d_v());
                dumpFieldStats("Home   uz", home.d_w());
            }
        }

        // 近傍壁面セルの |u| を定期採取（AMR 有効時の精度確認用）
        if(wallMetricsEvery > 0 && !nearWallFluid.empty() && (simStep % wallMetricsEvery) == 0){
            launch_speed(legacy.d_u(), legacy.d_v(), legacy.d_w(), d_speedLegacy, numCells);
            launch_speed(home.d_u(), home.d_v(), home.d_w(), d_speedHome, numCells);
            hSpeedLegacyHost.resize(numCells);
            hSpeedHomeHost.resize(numCells);
            cudaMemcpy(hSpeedLegacyHost.data(), d_speedLegacy, sizeof(float)*numCells, cudaMemcpyDeviceToHost);
            cudaMemcpy(hSpeedHomeHost.data(),   d_speedHome,   sizeof(float)*numCells,   cudaMemcpyDeviceToHost);
            double sumL=0.0, sumH=0.0;
            float maxL=0.0f, maxH=0.0f;
            for(int idx : nearWallFluid){
                float sl = hSpeedLegacyHost[idx];
                float sh = hSpeedHomeHost[idx];
                sumL += sl; sumH += sh;
                maxL = std::max(maxL, std::abs(sl));
                maxH = std::max(maxH, std::abs(sh));
            }
            double count = static_cast<double>(nearWallFluid.size());
            std::printf("[wall] step %d  |u|_mean Legacy=%.3e Home=%.3e  max=%.3e / %.3e  cells=%zu\n",
                        simStep, sumL/count, sumH/count, maxL, maxH, nearWallFluid.size());
        }

        // 表示するスカラー場の選択
if (mode == Display::Pressure) {
    const float* rhoSrc = nullptr;
    if(solverView == SolverView::Legacy)      rhoSrc = legacy.d_rho();
    else if(solverView == SolverView::Home)   rhoSrc = home.d_rho();
    else if(solverView == SolverView::Hybrid) rhoSrc = hybrid.d_rho();
    else /* Compare */                        rhoSrc = home.d_rho();
    launch_shift(rhoSrc, -1.0f, d_tmp, numCells);   // ρ → ρ-1
    vis.uploadScalar(d_tmp);
}

        else if(mode == Display::Speed){
            if(solverView == SolverView::Legacy){
                launch_speed(legacy.d_u(), legacy.d_v(), legacy.d_w(), d_speedLegacy, numCells);
                vis.uploadScalar(d_speedLegacy);
            }else if(solverView == SolverView::Home || solverView == SolverView::Compare){
                launch_speed(home.d_u(), home.d_v(), home.d_w(), d_speedHome, numCells);
                vis.uploadScalar(d_speedHome);
            }else{ // Hybrid
                launch_speed(hybrid.d_u(), hybrid.d_v(), hybrid.d_w(), d_speedHome, numCells);
                vis.uploadScalar(d_speedHome);
            }
        }
        else { // Error (Compare default: HOME - Legacy)
            if(solverView == SolverView::Hybrid){
                launch_diff(hybrid.d_rho(), legacy.d_rho(), d_diffRho, numCells);
            }else{
                launch_diff(home.d_rho(), legacy.d_rho(), d_diffRho, numCells);
            }
            vis.uploadScalar(d_diffRho);
        }

        // ==== マウス操作（左:回転 / 右:ズーム） ====
{
    double cx=0.0, cy=0.0;
    glfwGetCursorPos(win.handle(), &cx, &cy);

    // 左ボタン：回転
    int lmb = glfwGetMouseButton(win.handle(), GLFW_MOUSE_BUTTON_LEFT);
    if(lmb == GLFW_PRESS){
        if(!orb.rotating){ orb.rotating = true; orb.lastX = cx; orb.lastY = cy; }
        double dx = cx - orb.lastX;
        double dy = cy - orb.lastY;
        orb.lastX = cx; orb.lastY = cy;

        const float sens = 0.005f; // 回転感度
        orb.yaw   -= (float)dx * sens;
        orb.pitch -= (float)dy * sens;

        // ピッチ制限（真上/真下付近で発散しないように）
        if(orb.pitch >  1.55f) orb.pitch =  1.55f;
        if(orb.pitch < -1.55f) orb.pitch = -1.55f;
    }else{
        orb.rotating = false;
    }

    // 右ボタン：ズーム（距離を変える）
    int rmb = glfwGetMouseButton(win.handle(), GLFW_MOUSE_BUTTON_RIGHT);
    if(rmb == GLFW_PRESS){
        if(!orb.dollying){ orb.dollying = true; orb.lastX = cx; orb.lastY = cy; }
        double dy = cy - orb.lastY;
        orb.lastY = cy;

        const float zsense = 0.01f; // ズーム感度
        orb.radius *= (1.0f + (float)dy * zsense);
        if(orb.radius < 0.2f) orb.radius = 0.2f;
        if(orb.radius > 10.0f) orb.radius = 10.0f;
    }else{
        orb.dollying = false;
    }

    // 軌道パラメータ → カメラ位置へ反映
    const float cp = std::cos(orb.pitch), sp = std::sin(orb.pitch);
    const float cyaw = std::cos(orb.yaw),  syaw = std::sin(orb.yaw);

    cam.pos = orb.target + glm::vec3(
        orb.radius * cp * cyaw,
        orb.radius * sp,
        orb.radius * cp * syaw
    );
    cam.target = orb.target; // 念のため再設定
}
// ==== マウス操作ここまで ====


        // 描画
        cam.update(win.width(), win.height());
glClearColor(0.02f,0.02f,0.03f,1.0f);
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

// --- 体積（ボリューム）を先に、深度無効で描画 ---
glDisable(GL_DEPTH_TEST);          // 追加
float scaleVis =
    // Error は差分が小さいのでゲインを高めに
    (mode == Display::Error) ? 2000.0f :
    (mode == Display::Speed) ? 150.0f :
                               1000.0f;   // Pressure(ρ-1)はゲイン低めから
float alphaCut = 0.15f; // この値未満のボクセルは透明にする
vis.renderVolume(cam.P, cam.V, cam.pos, 0.005f, scaleVis, alphaCut);

// --- メッシュを後から重ねる（深度有効） ---
if (showObstacle) {
    glEnable(GL_DEPTH_TEST);       // 追加
    glm::mat4 VP = cam.P * cam.V;
    float color[3] = { 1.0f, 0.5f, 0.1f }; // 目立つ色
    meshR.draw(&VP[0][0], color);
}
// --- トレーサの軌跡（Legacy: ライムグリーン, HOME: マゼンタ） ---
glm::mat4 MVP = cam.P * cam.V;
tracerR.draw(tracerLegacy, MVP, glm::vec3(0.4f, 1.0f, 0.2f));
tracerR.draw(tracerHome,   MVP, glm::vec3(1.0f, 0.2f, 1.0f));

        glfwPollEvents();
        win.swap();
    }

    // 後片付け
    cudaFree(d_tmp);
    cudaFree(d_speedLegacy);
    cudaFree(d_speedHome);
    cudaFree(d_diffRho);
    return 0;
}
