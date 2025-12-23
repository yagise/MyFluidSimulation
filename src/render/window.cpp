
#define GLFW_INCLUDE_NONE
#include "window.hpp"
#include <stdexcept> 
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// summary: h_ の処理を行う
// param w_: 入力パラメータ
// param w_: 入力パラメータ
// param h_: 入力パラメータ
// return: 戻り値
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

// summary: glfwWindowShouldClose の処理を行う
// param: なし
// return: 戻り値
bool Window::shouldClose() const { return glfwWindowShouldClose(handle_); }
// summary: glfwSwapBuffers の処理を行う
// param: なし
// return: 戻り値
void Window::swap() const        { glfwSwapBuffers(handle_); }
Window::~Window()                { glfwTerminate(); }
