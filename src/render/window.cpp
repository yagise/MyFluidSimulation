
#define GLFW_INCLUDE_NONE
#include "window.hpp"
#include <stdexcept> 
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// GLFW 初期化と GLAD ロードまでを行う。以降のレンダラはこのコンテキストを前提にする。
Window::Window(int w_, int h_) : w_(w_), h_(h_) {
    if (!glfwInit()) throw std::runtime_error("glfwInit failed");
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    handle_ = glfwCreateWindow(w_, h_, "LBM", nullptr, nullptr);
    if (!handle_) throw std::runtime_error("create window failed");
    glfwMakeContextCurrent(handle_);

    if (!gladLoadGL())
        throw std::runtime_error("glad init failed");

    glViewport(0, 0, w_, h_);
}
// ウィンドウが閉じるべきかを問い合わせる
bool Window::shouldClose() const { return glfwWindowShouldClose(handle_); }
// バックバッファをフロントへスワップする
void Window::swap() const        { glfwSwapBuffers(handle_); }
// GLFW 終了処理（コンテキスト破棄）
Window::~Window()                { glfwTerminate(); }
