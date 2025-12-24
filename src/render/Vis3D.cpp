
#include "Vis3D.hpp"
#include <stdexcept>
#include <glm/gtc/matrix_inverse.hpp>

static const float quad[30] = {
   -1,-1,0, 0,0,   1,-1,0, 1,0,   1,1,0, 1,1,
   -1,-1,0, 0,0,   1,1,0, 1,1,  -1,1,0, 0,1
};
// シェーダをコンパイルしてエラー時に例外を投げる
static GLuint compile_shader(GLenum type, const char* src){
    GLuint s = glCreateShader(type);
    glShaderSource(s,1,&src,nullptr);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
    if(!ok){
        char log[512]; glGetShaderInfoLog(s,512,nullptr,log);
        throw std::runtime_error(log);
    }
    return s;
}

static const char* vsrc =
    "#version 330 core\n"
    "layout(location=0) in vec3 pos;"
    "layout(location=1) in vec2 uv;"
    "out vec2 vUV;"
    "void main(){ gl_Position = vec4(pos,1); vUV = uv; }";

static const char* fsrcVol =
    "#version 330 core\n"
    "in vec2 vUV; out vec4 frag;\n"
    "uniform sampler3D uTex;                                  \n"
    "uniform mat4      uInvPV;                                \n"
    "uniform vec3      uCamPos;                               \n"
    "uniform float     uStep;                                 \n"
    "uniform float     uScale;                                \n"
    "uniform float     uThresh;                               \n"
    "\n"
    "vec3 colormap(float t){                                  \n"
    "   t = clamp(t,0.0,1.0);                                 \n"
    "   if(t < 0.25) return mix(vec3(0.02,0.02,0.2), vec3(0.0,0.7,1.0), t*4.0);\n"
    "   if(t < 0.50) return mix(vec3(0.0,0.7,1.0), vec3(0.0,1.0,0.3), (t-0.25)*4.0);\n"
    "   if(t < 0.75) return mix(vec3(0.0,1.0,0.3), vec3(1.0,0.9,0.0), (t-0.50)*4.0);\n"
    "   return mix(vec3(1.0,0.9,0.0), vec3(1.0,0.1,0.0), (t-0.75)*4.0);\n"
    "}\n"
    "\n"
    "vec4 sampleVol(vec3 p){                                  \n"
    "   float s = texture(uTex,p).r;                          \n"
    "   float scaled = s * uScale;                            \n"
    "   float tCol = 0.5 + 0.5 * clamp(scaled, -1.0, 1.0);    \n"
    "   float tMag = clamp(abs(scaled), 0.0, 1.0);            \n"
    "   if(tMag < uThresh) return vec4(0.0,0.0,0.0,0.0);      \n"
    "   float a = smoothstep(uThresh,1.0,tMag);               \n"
    "   vec3 col = colormap(tCol);                            \n"
    "   return vec4(col, a);                                  \n"
    "}\n"
    "void main(){                                             \n"
    "   vec4 ndc = vec4(vUV*2.0-1.0, 0.0, 1.0);               \n"
    "   vec4 wPos = uInvPV * ndc;                             \n"
    "   wPos /= wPos.w;                                       \n"
    "   vec3 dir = normalize(wPos.xyz - uCamPos);             \n"
    "   vec3 t0 = (vec3(0)-uCamPos)/dir;                      \n"
    "   vec3 t1 = (vec3(1)-uCamPos)/dir;                      \n"
    "   float tNear = max(max(min(t0.x,t1.x),min(t0.y,t1.y)), min(t0.z,t1.z));\n"
    "   float tFar  = min(min(max(t0.x,t1.x),max(t0.y,t1.y)), max(t0.z,t1.z));\n"
    "   if(tFar<0.0 || tNear>tFar) discard;                   \n"
    "   tNear = max(tNear, 0.0);                              \n"
    "   vec3 pos = uCamPos + dir * tNear;                     \n"
    "   float t = tNear; vec4 acc = vec4(0);                  \n"
    "   while(t < tFar && acc.a < 0.95){                      \n"
    "       vec4 s = sampleVol(pos);                          \n"
    "       acc.rgb += s.rgb * (1.0 - acc.a);                 \n"
    "       acc.a   += s.a   * (1.0 - acc.a);                 \n"
    "       pos += dir * uStep; t += uStep;                   \n"
    "   }                                                     \n"
    "   frag = vec4(acc.rgb, 1.0);                            \n"
    "}";
// 3D テクスチャとフルスクリーンクアッドのセットアップ
Vis3D::Vis3D(int nx,int ny,int nz):Nx(nx),Ny(ny),Nz(nz){
    glGenTextures(1,&tex3d);
    glBindTexture(GL_TEXTURE_3D,tex3d);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, Nx, Ny, Nz, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_WRAP_R,GL_CLAMP_TO_EDGE);

    glGenVertexArrays(1,&vao);
    glGenBuffers(1,&vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER,vbo);
    glBufferData(GL_ARRAY_BUFFER,sizeof(quad),quad,GL_STATIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,5*sizeof(float),(void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,5*sizeof(float),(void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);

    GLuint vs   = compile_shader(GL_VERTEX_SHADER,  vsrc);
    GLuint fsV  = compile_shader(GL_FRAGMENT_SHADER,fsrcVol);
    progVol     = glCreateProgram();
    glAttachShader(progVol,vs);
    glAttachShader(progVol,fsV);
    glLinkProgram(progVol);
    glDeleteShader(vs); glDeleteShader(fsV);
    GLint ok; glGetProgramiv(progVol,GL_LINK_STATUS,&ok);
    if(!ok){ char log[256]; glGetProgramInfoLog(progVol,256,nullptr,log); throw std::runtime_error(log); }

    glUseProgram(progVol);
    glUniform1i(glGetUniformLocation(progVol,"uTex"),0);
}
// ホスト側のバッファが必要サイズか確認・確保する
void Vis3D::ensureHostBuffers(){
    std::size_t cells = (std::size_t)Nx*Ny*Nz;
    if(h_scalar.size()!=cells) h_scalar.resize(cells);
}
// デバイスのスカラー場を読み戻し、3D テクスチャに転送する
void Vis3D::uploadScalar(const float* d_field){
    ensureHostBuffers();
    std::size_t cells = (std::size_t)Nx*Ny*Nz;
    cudaMemcpy(h_scalar.data(), d_field, cells*sizeof(float), cudaMemcpyDeviceToHost);
    glBindTexture(GL_TEXTURE_3D,tex3d);
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0,0,0, Nx,Ny,Nz, GL_RED, GL_FLOAT, h_scalar.data());
}
// 逆 PV 行列からレイを作り、閾値付きのレイマーチで可視化する
void Vis3D::renderVolume(const glm::mat4& P, const glm::mat4& V,
                         const glm::vec3& camPos, float step, float scale, float threshold){
    glm::mat4 invPV = glm::inverse(P*V);
    glUseProgram(progVol);
    glUniformMatrix4fv(glGetUniformLocation(progVol,"uInvPV"),1,GL_FALSE,&invPV[0][0]);
    glUniform3fv(glGetUniformLocation(progVol,"uCamPos"),1,&camPos[0]);
    glUniform1f(glGetUniformLocation(progVol,"uStep"), step);
    glUniform1f(glGetUniformLocation(progVol,"uScale"), scale);
    glUniform1f(glGetUniformLocation(progVol,"uThresh"), threshold);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES,0,6);
}
