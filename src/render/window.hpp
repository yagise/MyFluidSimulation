
#pragma once
#include <GLFW/glfw3.h>
class Window {
public:
    // summary: Window の処理を行う
    // param w: 入力パラメータ
    // param h: 入力パラメータ
    // return: 戻り値
    Window(int w, int h);
    // summary: ~Window の処理を行う
    // param: なし
    // return: 戻り値
    ~Window();
    // summary: shouldClose の処理を行う
    // param: なし
    // return: 戻り値
    bool shouldClose() const;
    // summary: swap の処理を行う
    // param: なし
    // return: なし
    void swap() const;
    // summary: handle の処理を行う
    // param: なし
    // return: 戻り値
    GLFWwindow* handle() const { return handle_; }
    // summary: width の処理を行う
    // param: なし
    // return: 戻り値
    int width() const { return w_; }
    // summary: height の処理を行う
    // param: なし
    // return: 戻り値
    int height() const { return h_; }
private:
    int w_, h_;
    GLFWwindow* handle_ = nullptr;
};
