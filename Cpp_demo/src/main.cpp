#include <casadi/casadi.hpp>

#include <imgui.h>
#include <implot.h>
#include <GLFW/glfw3.h>

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <iomanip>

static void glfw_error_callback( int error, const char* description ) {
    std::fprintf( stderr, "GLFW Error %d: %s\n", error, description );
}

// 一次性跑完 MPC 仿真，把数据写入 vectors
static bool run_mpc_simulation( casadi::Opti&     opti,                                                                  //
                                const casadi::MX& X, const casadi::MX& U, const casadi::MX& x0, const casadi::MX& xref,  //
                                const casadi::DM& A, const casadi::DM& B, casadi::DM x, const casadi::DM& x_ref,         //
                                int steps, int N, int nx, int nu,                                                        //
                                std::vector< double >& ts, std::vector< double >& xs0, std::vector< double >& xs1,       //
                                std::vector< double >& us0, std::string& last_err ) {
    using namespace casadi;

    ts.clear();  // 清除
    xs0.clear();
    xs1.clear();
    us0.clear();
    ts.reserve( steps );  // 内存预分配
    xs0.reserve( steps );
    xs1.reserve( steps );
    us0.reserve( steps );

    // 初始猜测有助于 SQP（可保留）
    opti.set_initial( X, DM::zeros( nx, N + 1 ) );
    opti.set_initial( U, DM::zeros( nu, N ) );

    // 计时：记录上次到达此处的时间点，用于计算间隔
    std::chrono::steady_clock::time_point last_time = std::chrono::steady_clock::now();

    for ( int t_step = 0; t_step < steps; ++t_step ) {
        opti.set_value( x0, x );
        opti.set_value( xref, x_ref );

        try {
            auto sol = opti.solve();

            DM u0 = sol.value( U( Slice(), 0 ) );

            const double pos = ( double )x( 0 );
            const double vel = ( double )x( 1 );
            const double u   = ( double )u0( 0 );

            // 计算自上次打印到此处的时间间隔（毫秒）并打印
            auto   now        = std::chrono::steady_clock::now();
            double elapsed_ms = std::chrono::duration_cast< std::chrono::duration< double, std::milli > >( now - last_time ).count();
            // 使用制表符对齐各列输出，并固定 3 位小数
            std::cout << std::fixed << std::setprecision( 3 );
            std::cout << "t=" << t_step << '\t' << "x=[" << pos << ", " << vel << "]" << '\t' << "u0=" << u << '\t' << "dt_ms=" << elapsed_ms << std::endl;
            // 恢复默认浮点格式
            std::cout.unsetf( std::ios::fixed );
            std::cout << std::setprecision( 6 );
            last_time = now;

            // 记录数据
            ts.push_back( ( double )t_step );
            xs0.push_back( pos );
            xs1.push_back( vel );
            us0.push_back( u );

            // 更新系统状态
            x = mtimes( A, x ) + mtimes( B, u0 );

            // Warm start：shift solution（与你原逻辑一致）
            DM U_sol = sol.value( U );
            DM X_sol = sol.value( X );

            DM U_guess = DM::zeros( nu, N );
            for ( int k = 0; k < N - 1; ++k )
                U_guess( Slice(), k ) = U_sol( Slice(), k + 1 );
            U_guess( Slice(), N - 1 ) = DM::zeros( nu, 1 );

            DM X_guess = DM::zeros( nx, N + 1 );
            for ( int k = 0; k < N; ++k )
                X_guess( Slice(), k ) = X_sol( Slice(), k + 1 );
            X_guess( Slice(), N ) = X_sol( Slice(), N );

            opti.set_initial( U, U_guess );
            opti.set_initial( X, X_guess );

            last_err.clear();
        }
        catch ( std::exception& e ) {
            last_err = e.what();
            std::cerr << "Solve failed at t=" << t_step << " : " << last_err << std::endl;
            return false;
        }
    }

    return true;
}

int main() {
    // -----------------------
    // GLFW + OpenGL context
    // -----------------------
    // 设置错误回调（当 GLFW 内部出错时被调用）
    glfwSetErrorCallback( glfw_error_callback );
    // 初始化 GLFW，失败则退出
    if ( !glfwInit() )
        return 1;

    // OpenGL 版本按你环境调整；一般 3.3 Core 没问题
    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );                  // 主版本号
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 );                  // 次版本号
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );  // 使用 Core profile
#ifdef __APPLE__
    glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );  // MacOS 兼容性设置
#endif

    using namespace casadi;

    // -----------------------
    // MPC problem dimensions
    // -----------------------
    const int nx    = 2;   // 状态维度（位置 + 速度）
    const int nu    = 1;   // 控制输入维度（加速度/力）
    const int N     = 20;  // 预测时域步数
    const int steps = 30;  // 仿真步数（离线生成的时间点数量）

    const double dt = 0.1;                  // 采样时间间隔
    DM           A  = DM::zeros( nx, nx );  // 状态转移矩阵
    A( 0, 0 )       = 1;
    A( 0, 1 )       = dt;
    A( 1, 0 )       = 0;
    A( 1, 1 )       = 1;

    DM B      = DM::zeros( nx, nu );  // 控制输入矩阵
    B( 0, 0 ) = 0.5 * dt * dt;        // 位移受加速度积分影响
    B( 1, 0 ) = dt;                   // 速度受加速度影响

    // 状态权重矩阵 Q（阶段成本）
    DM Q      = DM::zeros( nx, nx );
    Q( 0, 0 ) = 10.0;  // 位置误差权重较高
    Q( 1, 1 ) = 1.0;   // 速度误差权重较低

    // 终端权重矩阵 QN（终端成本）
    DM QN      = DM::zeros( nx, nx );
    QN( 0, 0 ) = 20.0;
    QN( 1, 1 ) = 2.0;

    // 控制输入权重 R（耗用控制力度的惩罚）
    DM R      = DM::zeros( nu, nu );
    R( 0, 0 ) = 0.1;

    const double umax = 2.0;  // 控制饱和上限（|u| <= umax）

    // -----------------------
    // Build Opti MPC
    // -----------------------
    Opti opti;  // CasADi 的优化器容器（构建优化问题）

    MX X = opti.variable( nx, N + 1 );  // 优化变量：状态轨迹 (nx x (N+1))
    MX U = opti.variable( nu, N );      // 优化变量：控制序列 (nu x N)

    MX x0   = opti.parameter( nx, 1 );  // 参数：当前时刻初始状态（作为约束输入）
    MX xref = opti.parameter( nx, 1 );  // 参数：期望参考状态

    // 初始状态约束：X[:,0] == x0
    opti.subject_to( X( Slice(), 0 ) == x0 );

    // 动力学约束：X[:,k+1] = A*X[:,k] + B*U[:,k]
    for ( int k = 0; k < N; ++k ) {
        MX xk  = X( Slice(), k );
        MX uk  = U( Slice(), k );
        MX xkp = X( Slice(), k + 1 );
        // 使用矩阵乘法 mtimes 构建线性动力学约束
        opti.subject_to( xkp == mtimes( A, xk ) + mtimes( B, uk ) );
    }

    // 控制量饱和约束：-umax <= U <= umax（对所有时刻元素逐项约束）
    opti.subject_to( -umax <= U <= umax );

    // 构造二次成本函数（轨迹误差和控制用量）
    MX cost = 0;
    for ( int k = 0; k < N; ++k ) {
        MX e = X( Slice(), k ) - xref;                     // 当前时刻状态误差
        cost += mtimes( mtimes( transpose( e ), Q ), e );  // e'Qe 运行代价
        MX uk = U( Slice(), k );
        cost += mtimes( mtimes( transpose( uk ), R ), uk );  // u'Ru 控制代价
    }
    MX eN = X( Slice(), N ) - xref;                       // 终端误差
    cost += mtimes( mtimes( transpose( eN ), QN ), eN );  // 终端成本
    opti.minimize( cost );                                // 设定优化目标

    // 设置求解器选项（使用 SQP 方法 + 内部 QP 求解器 qrqp）
    Dict nlp_opts;
    nlp_opts[ "qpsol" ]           = "qrqp";
    nlp_opts[ "qpsol_options" ]   = Dict{ { "print_iter", false }, { "print_header", false } };
    nlp_opts[ "max_iter" ]        = 30;
    nlp_opts[ "print_header" ]    = false;
    nlp_opts[ "print_iteration" ] = false;
    nlp_opts[ "print_time" ]      = false;
    nlp_opts[ "print_status" ]    = false;

    // 将配置传给 Opti，选择求解器 "sqpmethod"
    opti.solver( "sqpmethod", nlp_opts );

    // -----------------------
    // Initial state / reference
    // -----------------------
    DM x   = DM::zeros( nx, 1 );  // 当前真实状态（用于仿真起点）
    x( 0 ) = 0.0;                 // 位置
    x( 1 ) = 0.0;                 // 速度

    DM x_ref   = DM::zeros( nx, 1 );  // 目标参考状态
    x_ref( 0 ) = 1.0;                 // 目标位置
    x_ref( 1 ) = 0.0;                 // 目标速度

    // -----------------------
    // 1) 先生成数据（离线跑完）
    // -----------------------
    std::vector< double > ts, xs0, xs1, us0;  // 时间轴、位置序列、速度序列、控制序列
    std::string           last_err;           // 记录最后的错误信息（若求解失败）

    // 调用仿真函数，传入 Opti 对象与所有必要参数
    // run_mpc_simulation 会在离线仿真中反复求解 MPC 并填充 ts,xs0,xs1,us0
    bool ok = run_mpc_simulation( opti, X, U, x0, xref, A, B, x, x_ref, steps, N, nx, nu, ts, xs0, xs1, us0, last_err );

    // 如果求解失败，你也可以选择直接退出
    // if (!ok) return 1;

    // -----------------------
    // 2) 再初始化 GUI，仅绘制
    // -----------------------
    // 重新设置错误回调并初始化 GLFW（之前可能在仿真中已改动）
    glfwSetErrorCallback( glfw_error_callback );
    if ( !glfwInit() )
        return 1;

    // 再次设置 OpenGL context hints
    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 );
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );
#ifdef __APPLE__
    glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );
#endif

    // 创建窗口（1200x800）用于绘图
    GLFWwindow* window = glfwCreateWindow( 1200, 800, "CasADi MPC + ImPlot (offline plot)", nullptr, nullptr );
    if ( !window ) {
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent( window );
    glfwSwapInterval( 1 );  // 启用垂直同步（1 = 开启）

    // 初始化 ImGui / ImPlot 上下文与后端绑定
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL( window, true );  // 将 ImGui 与 GLFW 绑定（处理输入）
    ImGui_ImplOpenGL3_Init( "#version 330" );      // 指定 GLSL 版本给 OpenGL3 后端

    const float u_sat = ( float )umax;  // 用于绘图显示上下限

    // 主循环：只是绘制离线生成的序列，不进行在线控制
    while ( !glfwWindowShouldClose( window ) ) {
        glfwPollEvents();  // 处理窗口事件（输入、窗口关闭等）

        // 开始新一帧的 ImGui / ImPlot 绘制命令
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // 状态窗口：显示仿真点数与求解状态
        ImGui::Begin( "Status" );
        ImGui::Text( "Simulation points: %d", ( int )ts.size() );
        ImGui::Text( "Solve status: %s", ok ? "OK" : "FAILED" );
        if ( !ok && !last_err.empty() ) {
            ImGui::TextWrapped( "Error: %s", last_err.c_str() );  // 如果有错误，显示错误信息
        }
        ImGui::End();

        // 绘制 ImPlot 图表区域
        ImGui::Begin( "ImPlot" );

        // 状态曲线（位置、速度）
        if ( ImPlot::BeginPlot( "States", ImVec2( -1, 320 ) ) ) {
            ImPlot::SetupAxes( "t", "x" );
            // 需要固定纵轴可在这里设置：
            // ImPlot::SetupAxisLimits(ImAxis_Y1, -2.0, 2.0, ImGuiCond_Always);

            if ( !ts.empty() ) {
                ImPlot::PlotLine( "pos", ts.data(), xs0.data(), ( int )ts.size() );  // 位置
                ImPlot::PlotLine( "vel", ts.data(), xs1.data(), ( int )ts.size() );  // 速度
            }
            ImPlot::EndPlot();
        }

        // 控制输入曲线
        if ( ImPlot::BeginPlot( "Control u0", ImVec2( -1, 260 ) ) ) {
            ImPlot::SetupAxes( "t", "u" );
            ImPlot::SetupAxisLimits( ImAxis_Y1, -3, 3, ImGuiCond_Always );  // 手动限制纵轴范围

            if ( !ts.empty() ) {
                ImPlot::PlotLine( "u0", ts.data(), us0.data(), ( int )ts.size() );  // 控制序列
            }

            // ----------- 红色水平直线：u = ±umax -----------
            if ( ts.size() >= 2 ) {

                double x_line[ 2 ] = { ts.front(), ts.back() };  // 水平线横向覆盖整个 x 轴范围

                double y_upper[ 2 ] = { u_sat, u_sat };    // 上界
                double y_lower[ 2 ] = { -u_sat, -u_sat };  // 下界

                // 红色线样式
                ImPlot::PushStyleColor( ImPlotCol_Line, ImVec4( 1.0f, 0.0f, 0.0f, 0.9f ) );
                ImPlot::PushStyleVar( ImPlotStyleVar_LineWeight, 0.5f );

                // 前缀 "##" 表示不在 legend 中显示标签
                ImPlot::PlotLine( "##u_max", x_line, y_upper, 2 );
                ImPlot::PlotLine( "##-u_max", x_line, y_lower, 2 );

                ImPlot::PopStyleVar();
                ImPlot::PopStyleColor();
            }
            // ---------------------------------------------

            ImPlot::EndPlot();
        }

        ImGui::End();

        // 渲染 ImGui 指令并显示到 GLFW 窗口
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize( window, &display_w, &display_h );  // 获取显示缓冲区大小（可能与窗口大小不同）
        glViewport( 0, 0, display_w, display_h );                  // 设置视口
        glClear( GL_COLOR_BUFFER_BIT );                            // 清除颜色缓冲

        ImGui_ImplOpenGL3_RenderDrawData( ImGui::GetDrawData() );  // 将 ImGui 绘制到屏幕
        glfwSwapBuffers( window );                                 // 交换前后缓冲显示本帧内容
    }

    // -----------------------
    // Cleanup
    // -----------------------
    ImGui_ImplOpenGL3_Shutdown();  // 清理 OpenGL3 后端资源
    ImGui_ImplGlfw_Shutdown();     // 清理 GLFW 后端资源
    ImPlot::DestroyContext();      // 销毁 ImPlot 上下文
    ImGui::DestroyContext();       // 销毁 ImGui 上下文
    glfwDestroyWindow( window );   // 销毁 GLFW 窗口
    glfwTerminate();               // 终止 GLFW

    return 0;
}
