
#include "MeshRenderer.hpp"
#include <stdexcept>

// summary: compile の処理を行う
// param type: 入力パラメータ
// param src: 入力パラメータ
// return: 戻り値
static GLuint compile(GLenum type, const char* src){
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
    "layout(location=0) in vec3 aPos; \n"
    "uniform mat4 uVP; \n"
    // summary: vec4 の処理を行う
    // param: なし
    // return: 戻り値
    "void main(){ gl_Position = uVP * vec4(aPos, 1.0); }"; 

static const char* fsrc =
    "#version 330 core\n"
    "out vec4 frag;\n"
    "uniform vec3 uColor;\n"
    "uniform float uAlpha;\n"
    // summary: vec4 の処理を行う
    // param: なし
    // return: 戻り値
    "void main(){ frag = vec4(uColor, uAlpha); }";

// summary: ~MeshRenderer の処理を行う
// param: なし
// return: 戻り値
MeshRenderer::~MeshRenderer(){
    if(ebo_) glDeleteBuffers(1,&ebo_);
    if(vbo_) glDeleteBuffers(1,&vbo_);
    if(vao_) glDeleteVertexArrays(1,&vao_);
    if(prog_) glDeleteProgram(prog_);
}

// summary: upload の処理を行う
// param positions: 入力パラメータ
// param indices: 入力パラメータ
// return: 戻り値
void MeshRenderer::upload(const std::vector<glm::vec3>& positions,
                          const std::vector<unsigned>& indices){
    if(!prog_){
        GLuint vs = compile(GL_VERTEX_SHADER, vsrc);
        GLuint fs = compile(GL_FRAGMENT_SHADER, fsrc);
        prog_ = glCreateProgram();
        glAttachShader(prog_,vs); glAttachShader(prog_,fs);
        glLinkProgram(prog_);
        glDeleteShader(vs); glDeleteShader(fs);
    }
    if(!vao_){ glGenVertexArrays(1,&vao_); glGenBuffers(1,&vbo_); glGenBuffers(1,&ebo_); }
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER,vbo_);
    glBufferData(GL_ARRAY_BUFFER, positions.size()*sizeof(glm::vec3), positions.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(glm::vec3),(void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size()*sizeof(unsigned), indices.data(), GL_STATIC_DRAW);
    indexCount_ = (GLsizei)indices.size();
}

// summary: 描画処理を行う
// param viewProj: 入力パラメータ
// param color: 入力パラメータ
// return: 戻り値
void MeshRenderer::draw(const float* viewProj, const float* color){
    if(!prog_ || indexCount_==0) return;
    glUseProgram(prog_);
    GLint locVP = glGetUniformLocation(prog_,"uVP");
    glUniformMatrix4fv(locVP,1,GL_FALSE,viewProj);
    GLint locC = glGetUniformLocation(prog_,"uColor");
    glUniform3fv(locC,1,color);
    GLint locA = glGetUniformLocation(prog_,"uAlpha");
    glUniform1f(locA, alpha_);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_CULL_FACE);
    glBindVertexArray(vao_);
    glDrawElements(GL_TRIANGLES, indexCount_, GL_UNSIGNED_INT, 0);
    glDisable(GL_BLEND);
}
