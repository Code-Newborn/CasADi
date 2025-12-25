#include <casadi/casadi.hpp>
#include <iostream>
#include <vector>

int main() {
    using namespace casadi;

    // -----------------------
    // MPC problem dimensions
    // -----------------------
    const int nx = 2;   // state dimension
    const int nu = 1;   // control dimension
    const int N  = 20;  // horizon length

    // -----------------------
    // 离散时间线性系统 (示例: 双积分器)
    // x = [pos; vel], u = [acc]
    // -----------------------
    const double dt = 0.1;
    DM           A  = DM::zeros( nx, nx );
    A( 0, 0 )       = 1;
    A( 0, 1 )       = dt;
    A( 1, 0 )       = 0;
    A( 1, 1 )       = 1;

    DM B      = DM::zeros( nx, nu );
    B( 0, 0 ) = 0.5 * dt * dt;
    B( 1, 0 ) = dt;

    // 权重
    DM Q      = DM::zeros( nx, nx );
    Q( 0, 0 ) = 10.0;
    Q( 1, 1 ) = 1.0;

    // 终端权重
    DM QN      = DM::zeros( nx, nx );
    QN( 0, 0 ) = 20.0;
    QN( 1, 1 ) = 2.0;

    // 控制权重
    DM R      = DM::zeros( nu, nu );
    R( 0, 0 ) = 0.1;

    const double umax = 2.0;

    // -----------------------
    // 利用 Opti 构建 MPC
    // -----------------------
    Opti opti;

    MX X = opti.variable( nx, N + 1 );  // states
    MX U = opti.variable( nu, N );      // inputs

    MX x0   = opti.parameter( nx, 1 );
    MX xref = opti.parameter( nx, 1 );

    // 初始状态约束
    opti.subject_to( X( Slice(), 0 ) == x0 );

    // 动力学约束
    for ( int k = 0; k < N; ++k ) {
        MX xk  = X( Slice(), k );
        MX uk  = U( Slice(), k );
        MX xkp = X( Slice(), k + 1 );
        opti.subject_to( xkp == mtimes( A, xk ) + mtimes( B, uk ) );
    }

    // 输入约束
    opti.subject_to( -umax <= U <= umax );

    // 构建目标函数
    MX cost = 0;
    for ( int k = 0; k < N; ++k ) {
        MX e = X( Slice(), k ) - xref;
        cost += mtimes( mtimes( transpose( e ), Q ), e );
        MX uk = U( Slice(), k );
        cost += mtimes( mtimes( transpose( uk ), R ), uk );
    }
    MX eN = X( Slice(), N ) - xref;
    cost += mtimes( mtimes( transpose( eN ), QN ), eN );

    opti.minimize( cost );  // 目标函数

    // -----------------------
    // 设置 SQP 求解器选项
    // -----------------------
    casadi::Dict nlp_opts;

    // 使用 QRQP 作为 QP 求解器
    nlp_opts[ "qpsol" ]         = "qrqp";
    nlp_opts[ "qpsol_options" ] = casadi::Dict{ { "print_iter", false }, { "print_header", false } };

    nlp_opts[ "max_iter" ]        = 30;
    nlp_opts[ "print_header" ]    = false;
    nlp_opts[ "print_iteration" ] = false;
    nlp_opts[ "print_time" ]      = false;

    // 关键：solver 名称放这里，而不是放在 nlp_opts 里
    opti.solver( "sqpmethod", nlp_opts );

    // -----------------------
    // Simulate MPC loop
    // -----------------------
    DM x   = DM::zeros( nx, 1 );
    x( 0 ) = 0.0;  // 初始位置
    x( 1 ) = 0.0;  // 初始速度

    DM x_ref   = DM::zeros( nx, 1 );
    x_ref( 0 ) = 1.0;  // 目标位置
    x_ref( 1 ) = 0.0;  // 目标速度

    // 初始猜测有助于 SQP
    opti.set_initial( X, DM::zeros( nx, N + 1 ) );
    opti.set_initial( U, DM::zeros( nu, N ) );

    // MPC 主循环
    for ( int t = 0; t < 30; ++t ) {
        opti.set_value( x0, x );
        opti.set_value( xref, x_ref );

        try {
            auto sol = opti.solve();

            // 提取并打印第一个控制输入
            DM u0 = sol.value( U( Slice(), 0 ) );
            std::cout << "t=" << t << "  x=[" << double( x( 0 ) ) << ", " << double( x( 1 ) ) << "]"
                      << "  u0=" << double( u0( 0 ) ) << std::endl;

            // 施加控制输入并更新状态
            x = mtimes( A, x ) + mtimes( B, u0 );

            // Warm start: shift solution (simple)
            DM U_sol = sol.value( U );
            DM X_sol = sol.value( X );

            // Shift guesses: U(:,k)=U(:,k+1), last = 0
            DM U_guess = DM::zeros( nu, N );
            for ( int k = 0; k < N - 1; ++k )
                U_guess( Slice(), k ) = U_sol( Slice(), k + 1 );
            U_guess( Slice(), N - 1 ) = DM::zeros( nu, 1 );

            // Shift X guesses: X(:,k)=X(:,k+1), last = last
            DM X_guess = DM::zeros( nx, N + 1 );
            for ( int k = 0; k < N; ++k )
                X_guess( Slice(), k ) = X_sol( Slice(), k + 1 );
            X_guess( Slice(), N ) = X_sol( Slice(), N );

            opti.set_initial( U, U_guess );
            opti.set_initial( X, X_guess );
        }
        catch ( std::exception& e )  // 捕获求解异常
        {
            std::cerr << "Solve failed at t=" << t << " : " << e.what() << std::endl;
            break;
        }
    }

    return 0;
}
