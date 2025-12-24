
#pragma once
#include <GLFW/glfw3.h>

// GLFW ベースの単純なウィンドウラッパー（コンテキスト作成とスワップだけを担当）
class Window {
public:
    // w×h のウィンドウを開き OpenGL コンテキストを有効化する
    Window(int w, int h);
    ~Window();

    bool shouldClose() const;
    void swap() const;

    GLFWwindow* handle() const { return handle_; }
    int width() const { return w_; }
    int height() const { return h_; }

private:
    int w_, h_;
    GLFWwindow* handle_ = nullptr;
};
