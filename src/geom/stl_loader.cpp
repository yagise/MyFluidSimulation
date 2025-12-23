#include "stl_loader.hpp"
#include <fstream>
#include <sstream>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <glm/glm.hpp> // vec3

// summary: vmin3 の処理を行う
// param a: 入力パラメータ
// param b: 入力パラメータ
// return: 戻り値
static inline glm::vec3 vmin3(const glm::vec3& a, const glm::vec3& b){
    return glm::vec3(std::min(a.x,b.x), std::min(a.y,b.y), std::min(a.z,b.z));
}
// summary: vmax3 の処理を行う
// param a: 入力パラメータ
// param b: 入力パラメータ
// return: 戻り値
static inline glm::vec3 vmax3(const glm::vec3& a, const glm::vec3& b){
    return glm::vec3(std::max(a.x,b.x), std::max(a.y,b.y), std::max(a.z,b.z));
}
// summary: 状態を更新する
// param mn: 入力パラメータ
// param mx: 入力パラメータ
// param p: 入力パラメータ
// return: なし
static inline void update_bb(glm::vec3& mn, glm::vec3& mx, const glm::vec3& p){
    mn = vmin3(mn, p);
    mx = vmax3(mx, p);
}

#pragma pack(push,1)
struct BinTri {
    float n[3];
    float v0[3];
    float v1[3];
    float v2[3];
    unsigned short attr;
};
#pragma pack(pop)

// summary: 引数や入力設定を解析する
// param is: 入力パラメータ
// param out: 入力パラメータ
// return: 戻り値
static bool parse_ascii(std::istream& is, TriangleMesh& out){
    out.positions.clear();
    out.indices.clear();
    out.bbmin = glm::vec3( std::numeric_limits<float>::max() );
    out.bbmax = glm::vec3( -std::numeric_limits<float>::max() );

    std::string line;
    std::vector<glm::vec3> tri; tri.reserve(3);

    while(std::getline(is, line)){
        std::istringstream ss(line);
        std::string tok; ss >> tok;
        if(tok == "vertex"){
            float x=0,y=0,z=0; ss >> x >> y >> z;
            tri.emplace_back(x,y,z);
            if(tri.size()==3){
                const unsigned base = static_cast<unsigned>(out.positions.size());
                out.positions.push_back(tri[0]);
                out.positions.push_back(tri[1]);
                out.positions.push_back(tri[2]);
                out.indices.push_back(base+0);
                out.indices.push_back(base+1);
                out.indices.push_back(base+2);
                update_bb(out.bbmin, out.bbmax, tri[0]);
                update_bb(out.bbmin, out.bbmax, tri[1]);
                update_bb(out.bbmin, out.bbmax, tri[2]);
                tri.clear();
            }
        }
    }
    return !out.indices.empty();
}

// summary: 引数や入力設定を解析する
// param is: 入力パラメータ
// param out: 入力パラメータ
// return: 戻り値
static bool parse_binary(std::istream& is, TriangleMesh& out){
    char header[80];
    if(!is.read(header,80)) return false;
    uint32_t ntri = 0;
    if(!is.read(reinterpret_cast<char*>(&ntri), 4)) return false;

    out.positions.clear();
    out.indices.clear();
    out.positions.reserve(ntri*3);
    out.indices.reserve(ntri*3);
    out.bbmin = glm::vec3( std::numeric_limits<float>::max() );
    out.bbmax = glm::vec3( -std::numeric_limits<float>::max() );

    for(uint32_t i=0;i<ntri;++i){
        BinTri bt;
        if(!is.read(reinterpret_cast<char*>(&bt), sizeof(BinTri))) return false;
        glm::vec3 v0(bt.v0[0], bt.v0[1], bt.v0[2]);
        glm::vec3 v1(bt.v1[0], bt.v1[1], bt.v1[2]);
        glm::vec3 v2(bt.v2[0], bt.v2[1], bt.v2[2]);

        const unsigned base = static_cast<unsigned>(out.positions.size());
        out.positions.push_back(v0);
        out.positions.push_back(v1);
        out.positions.push_back(v2);
        out.indices.push_back(base+0);
        out.indices.push_back(base+1);
        out.indices.push_back(base+2);

        update_bb(out.bbmin, out.bbmax, v0);
        update_bb(out.bbmin, out.bbmax, v1);
        update_bb(out.bbmin, out.bbmax, v2);
    }
    return true;
}

// summary: 入力データを読み込む
// param path: 入力パラメータ
// param out: 入力パラメータ
// return: 戻り値
bool load_stl(const std::string& path, TriangleMesh& out){
    std::ifstream f(path, std::ios::binary);
    if(!f) return false;

    // summary: read の処理を行う
    // param: なし
    // return: 戻り値
    char head[6] = {}; f.read(head,5);
    f.clear(); f.seekg(0);

    if(std::string(head) == "solid"){
        std::stringstream ss; ss << f.rdbuf();
        std::string s = ss.str();
        std::istringstream is(s);
        if(parse_ascii(is, out)) return true;

        // fall back to binary if ascii failed
        std::ifstream fb(path, std::ios::binary);
        if(!fb) return false;
        return parse_binary(fb, out);
    }

    // binary
    return parse_binary(f, out);
}

// summary: make_teardrop の処理を行う
// param nu: 入力パラメータ
// param nv: 入力パラメータ
// return: 戻り値
TriangleMesh make_teardrop(unsigned nu, unsigned nv){
    TriangleMesh m;
    m.bbmin = glm::vec3( std::numeric_limits<float>::max() );
    m.bbmax = glm::vec3( -std::numeric_limits<float>::max() );
    auto id = [nu](unsigned i,unsigned j){ return j*(nu+1)+i; };

    // parametric surface
    std::vector<glm::vec3> grid; grid.reserve((nu+1)*(nv+1));
    for(unsigned j=0;j<=nv;++j){
        float v = (float)j/nv;
        float theta = v*3.14159265f;
        float r = 0.5f*(1.0f - 0.5f*std::cos(theta));
        for(unsigned i=0;i<=nu;++i){
            float u = (float)i/nu;
            float phi = u*2.0f*3.14159265f;
            glm::vec3 p(r*std::cos(phi), r*std::sin(phi), v - 0.5f);
            grid.push_back(p);
            update_bb(m.bbmin, m.bbmax, p);
        }
    }

    // triangles (two per quad)
    for(unsigned j=0;j<nv;++j){
        for(unsigned i=0;i<nu;++i){
            glm::vec3 v0 = grid[id(i,  j)];
            glm::vec3 v1 = grid[id(i+1,j)];
            glm::vec3 v2 = grid[id(i,  j+1)];
            glm::vec3 v3 = grid[id(i+1,j+1)];
            unsigned base = (unsigned)m.positions.size();
            // summary: end の処理を行う
            // param end: 入力パラメータ
            // return: 戻り値
            m.positions.insert(m.positions.end(), {v0,v2,v1,  v1,v2,v3});
            // summary: end の処理を行う
            // param end: 入力パラメータ
            // return: 戻り値
            m.indices.insert(m.indices.end(), {base+0,base+1,base+2,  base+3,base+4,base+5});
        }
    }
    return m;
}

// summary: make_fan の処理を行う
// param blades: 入力パラメータ
// param radius: 入力パラメータ
// param hub: 入力パラメータ
// param thickness: 入力パラメータ
// return: 戻り値
TriangleMesh make_fan(unsigned blades, float radius, float hub, float thickness){
    TriangleMesh m;
    m.bbmin = glm::vec3( std::numeric_limits<float>::max() );
    m.bbmax = glm::vec3( -std::numeric_limits<float>::max() );

    const unsigned rimSeg = 64;
    // hub disk
    unsigned base = (unsigned)m.positions.size();
    m.positions.push_back(glm::vec3(0,0,0));
    for(unsigned i=0;i<rimSeg;++i){
        float ang = (float)i/rimSeg * 2.0f*3.14159265f;
        glm::vec3 p(hub*std::cos(ang), hub*std::sin(ang), 0);
        m.positions.push_back(p);
    }
    for(unsigned i=1;i<=rimSeg;++i){
        unsigned a = base+0, b = base+i, c = base+(i%rimSeg)+1;
        // summary: end の処理を行う
        // param end: 入力パラメータ
        // return: 戻り値
        m.indices.insert(m.indices.end(), {a,b,c});
    }

    // blades
    for(unsigned b=0;b<blades;++b){
        float ang = (float)b/blades * 2.0f*3.14159265f;
        float tilt = 0.3f;
        glm::vec3 p1(hub*std::cos(ang), hub*std::sin(ang), 0);
        glm::vec3 p2(radius*std::cos(ang+tilt), radius*std::sin(ang+tilt),  thickness);
        glm::vec3 p3(radius*std::cos(ang-tilt), radius*std::sin(ang-tilt), -thickness);
        unsigned baseB = (unsigned)m.positions.size();
        // summary: end の処理を行う
        // param end: 入力パラメータ
        // return: 戻り値
        m.positions.insert(m.positions.end(), {p1,p2,p3});
        // summary: end の処理を行う
        // param end: 入力パラメータ
        // return: 戻り値
        m.indices.insert(m.indices.end(), {baseB+0,baseB+1,baseB+2});
    }

    // bounds
    for(const auto& p : m.positions){ update_bb(m.bbmin, m.bbmax, p); }
    return m;
}
