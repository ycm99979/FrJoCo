// 공유 버퍼 (두 스레드가 충돌 없이 데이터를 주고받는 창구)
realtime_tools::RealtimeBuffer<std::vector<double>> rt_tau_ff_buffer_;

// ==========================================================
// [Thread A] 백그라운드 MPC 스레드 (100Hz)
// on_activate() 같은 곳에서 별도 스레드로 띄워둡니다.
// ==========================================================
void mpc_background_loop() {
    while (is_running) {
        std::vector<double> tau_ff(29, 0.0);
        
        // 무거운 궤적 최적화 연산 (예: 5ms ~ 10ms 소요)
        my_mpc_solver->compute_feedforward(current_state, tau_ff);
        
        // 계산된 값을 버퍼에 안전하게 덮어쓰기 (Non-rt -> RT 방향)
        rt_tau_ff_buffer_.writeFromNonRT(tau_ff);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // 100Hz 맞추기
    }
}

// ==========================================================
// [Thread B] ROS 2 제어 루프 (1000Hz) - 무조건 1ms 안에 끝남!
// ==========================================================
controller_interface::return_type update(const rclcpp::Time &time, const rclcpp::Duration &period) {
    // 1. MPC가 계산해둔 가장 최신 tau_ff를 즉시 읽어옴 (대기 시간 0)
    auto tau_ff_ptr = rt_tau_ff_buffer_.readFromRT();
    std::vector<double> tau_ff = *tau_ff_ptr;

    // 2. ONNX (TensorRT) 신경망 추론 (0.3ms 소요)
    // 잔차 학습 모델이 현재 상태의 미세한 오차를 보상할 tau_fb를 계산
    model_->forward(obs_buffer_.data());
    const float* nn_output = model_->getOutputs();

    // 3. 최종 토크 합성 및 인가
    for(int i = 0; i < 29; i++) {
        double tau_fb = nn_output[i] * action_scale;
        double tau_final = tau_ff[i] + tau_fb;
        
        // 모터에 토크 명령 전송
        command_interfaces_[i].set_value(tau_final);
    }
    return controller_interface::return_type::OK;
}