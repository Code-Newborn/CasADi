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

static void glfw_error_callback( int error, const char* description ) {
    std::fprintf( stderr, "GLFW Error %d: %s\n", error, description );
}

// 一次性跑完 MPC 仿真，把数据写入 vectors
static bool run_mpc_simulation( casadi::Opti& opti, const casadi::MX& X, const casadi::MX& U, const casadi::MX& x0, const casadi::MX& xref, const casadi::DM& A, const casadi::DM& B, casadi::DM x,
                                const casadi::DM& x_ref, int steps, int N, int nx, int nu, std::vector< double >& ts, std::vector< double >& xs0, std::vector< double >& xs1,
                                std::vector< double >& us0, std::string& last_err ) {
    using namespace casadi;

    ts.clear();
    xs0.clear();
    xs1.clear();
    us0.clear();
    ts.reserve( steps );
    xs0.reserve( steps );
    xs1.reserve( steps );
    us0.reserve( steps );

    // 初始猜测有助于 SQP（可保留）
    opti.set_initial( X, DM::zeros( nx, N + 1 ) );
    opti.set_initial( U, DM::zeros( nu, N ) );

    for ( int t_step = 0; t_step < steps; ++t_step ) {
        opti.set_value( x0, x );
        opti.set_value( xref, x_ref );

        try {
            auto sol = opti.solve();

            DM u0 = sol.value( U( Slice(), 0 ) );

            const double pos = ( double )x( 0 );
            const double vel = ( double )x( 1 );
            const double u   = ( double )u0( 0 );

            // 你原来的打印仍然可保留
            std::cout << "t=" << t_step << "  x=[" << pos << ", " << vel << "]"
                      << "  u0=" << u << std::endl;

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
    glfwSetErrorCallback( glfw_error_callback );
    if ( !glfwInit() )
        return 1;

    // OpenGL 版本按你环境调整；一般 3.3 Core 没问题
    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 );
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );
#ifdef __APPLE__
    glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );
#endif

    using namespace casadi;

    // -----------------------
    // MPC problem dimensions
    // -----------------------
    const int nx    = 2;
    const int nu    = 1;
    const int N     = 20;
    const int steps = 30;  // 原来 for (t<30)

    const double dt = 0.1;
    DM           A  = DM::zeros( nx, nx );
    A( 0, 0 )       = 1;
    A( 0, 1 )       = dt;
    A( 1, 0 )       = 0;
    A( 1, 1 )       = 1;

    DM B      = DM::zeros( nx, nu );
    B( 0, 0 ) = 0.5 * dt * dt;
    B( 1, 0 ) = dt;

    DM Q      = DM::zeros( nx, nx );
    Q( 0, 0 ) = 10.0;
    Q( 1, 1 ) = 1.0;

    DM QN      = DM::zeros( nx, nx );
    QN( 0, 0 ) = 20.0;
    QN( 1, 1 ) = 2.0;

    DM R      = DM::zeros( nu, nu );
    R( 0, 0 ) = 0.1;

    const double umax = 2.0;

    // -----------------------
    // Build Opti MPC
    // -----------------------
    Opti opti;

    MX X = opti.variable( nx, N + 1 );
    MX U = opti.variable( nu, N );

    MX x0   = opti.parameter( nx, 1 );
    MX xref = opti.parameter( nx, 1 );

    opti.subject_to( X( Slice(), 0 ) == x0 );

    for ( int k = 0; k < N; ++k ) {
        MX xk  = X( Slice(), k );
        MX uk  = U( Slice(), k );
        MX xkp = X( Slice(), k + 1 );
        opti.subject_to( xkp == mtimes( A, xk ) + mtimes( B, uk ) );
    }

    opti.subject_to( -umax <= U <= umax );

    MX cost = 0;
    for ( int k = 0; k < N; ++k ) {
        MX e = X( Slice(), k ) - xref;
        cost += mtimes( mtimes( transpose( e ), Q ), e );
        MX uk = U( Slice(), k );
        cost += mtimes( mtimes( transpose( uk ), R ), uk );
    }
    MX eN = X( Slice(), N ) - xref;
    cost += mtimes( mtimes( transpose( eN ), QN ), eN );
    opti.minimize( cost );

    Dict nlp_opts;
    nlp_opts[ "qpsol" ]           = "qrqp";
    nlp_opts[ "qpsol_options" ]   = Dict{ { "print_iter", false }, { "print_header", false } };
    nlp_opts[ "max_iter" ]        = 30;
    nlp_opts[ "print_header" ]    = false;
    nlp_opts[ "print_iteration" ] = false;
    nlp_opts[ "print_time" ]      = false;
    nlp_opts[ "print_status" ]    = false;

    opti.solver( "sqpmethod", nlp_opts );

    // -----------------------
    // Initial state / reference
    // -----------------------
    DM x   = DM::zeros( nx, 1 );
    x( 0 ) = 0.0;
    x( 1 ) = 0.0;

    DM x_ref   = DM::zeros( nx, 1 );
    x_ref( 0 ) = 1.0;
    x_ref( 1 ) = 0.0;

    // -----------------------
    // 1) 先生成数据（离线跑完）
    // -----------------------
    std::vector< double > ts, xs0, xs1, us0;
    std::string           last_err;

    bool ok = run_mpc_simulation( opti, X, U, x0, xref, A, B, x, x_ref, steps, N, nx, nu, ts, xs0, xs1, us0, last_err );

    // 如果求解失败，你也可以选择直接退出
    // if (!ok) return 1;

    // -----------------------
    // 2) 再初始化 GUI，仅绘制
    // -----------------------
    glfwSetErrorCallback( glfw_error_callback );
    if ( !glfwInit() )
        return 1;

    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 );
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );
#ifdef __APPLE__
    glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );
#endif

    GLFWwindow* window = glfwCreateWindow( 1200, 800, "CasADi MPC + ImPlot (offline plot)", nullptr, nullptr );
    if ( !window ) {
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent( window );
    glfwSwapInterval( 1 );

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL( window, true );
    ImGui_ImplOpenGL3_Init( "#version 330" );

    const float u_sat = ( float )umax;

    while ( !glfwWindowShouldClose( window ) ) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin( "Status" );
        ImGui::Text( "Simulation points: %d", ( int )ts.size() );
        ImGui::Text( "Solve status: %s", ok ? "OK" : "FAILED" );
        if ( !ok && !last_err.empty() ) {
            ImGui::TextWrapped( "Error: %s", last_err.c_str() );
        }
        ImGui::End();

        ImGui::Begin( "ImPlot" );

        if ( ImPlot::BeginPlot( "States", ImVec2( -1, 320 ) ) ) {
            ImPlot::SetupAxes( "t", "x" );
            // 需要固定纵轴可在这里设置：
            // ImPlot::SetupAxisLimits(ImAxis_Y1, -2.0, 2.0, ImGuiCond_Always);

            if ( !ts.empty() ) {
                ImPlot::PlotLine( "pos", ts.data(), xs0.data(), ( int )ts.size() );
                ImPlot::PlotLine( "vel", ts.data(), xs1.data(), ( int )ts.size() );
            }
            ImPlot::EndPlot();
        }

        if ( ImPlot::BeginPlot( "Control u0", ImVec2( -1, 260 ) ) ) {
            ImPlot::SetupAxes( "t", "u" );
            ImPlot::SetupAxisLimits( ImAxis_Y1, -3, 3, ImGuiCond_Always );

            if ( !ts.empty() ) {
                ImPlot::PlotLine( "u0", ts.data(), us0.data(), ( int )ts.size() );
            }

            // ----------- 红色水平直线：u = ±umax -----------
            if ( ts.size() >= 2 ) {

                double x_line[ 2 ] = { ts.front(), ts.back() };

                double y_upper[ 2 ] = { u_sat, u_sat };
                double y_lower[ 2 ] = { -u_sat, -u_sat };

                // 红色
                ImPlot::PushStyleColor( ImPlotCol_Line, ImVec4( 1.0f, 0.0f, 0.0f, 0.9f ) );
                ImPlot::PushStyleVar( ImPlotStyleVar_LineWeight, 0.5f );

                // ## 前缀：不显示在 legend
                ImPlot::PlotLine( "##u_max", x_line, y_upper, 2 );
                ImPlot::PlotLine( "##-u_max", x_line, y_lower, 2 );

                ImPlot::PopStyleVar();
                ImPlot::PopStyleColor();
            }
            // ---------------------------------------------

            ImPlot::EndPlot();
        }

        ImGui::End();

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize( window, &display_w, &display_h );
        glViewport( 0, 0, display_w, display_h );
        glClear( GL_COLOR_BUFFER_BIT );

        ImGui_ImplOpenGL3_RenderDrawData( ImGui::GetDrawData() );
        glfwSwapBuffers( window );
    }

    // -----------------------
    // Cleanup
    // -----------------------
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow( window );
    glfwTerminate();

    return 0;
}
