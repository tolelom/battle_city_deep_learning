import os, datetime, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from battle_ssafy_env import BattleSsafyEnv
from tqdm import tqdm

# 환경 상수들 (BattleSsafyEnv에서 가져옴)
G, R, W, F, X, E, S, T, H = 0, 1, 2, 3, 4, 5, 6, 7, 8

class TqdmCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.pbar = None
        self.start_num = 0
        self.goal = 0

    def _on_training_start(self) -> None:
        self.start_num = int(self.model.num_timesteps)
        to_add = int(getattr(self.model, "_total_timesteps", 0))
        self.goal = self.start_num + to_add

        self.pbar = tqdm(
            total=self.goal,
            initial=self.start_num,
            desc="Training",
            unit="steps",
            dynamic_ncols=True,
            mininterval=0.2,
            leave=True,
        )

    def _on_step(self) -> bool:
        if self.pbar:
            curr = int(self.model.num_timesteps)
            if curr > self.pbar.n:
                self.pbar.update(curr - self.pbar.n)
        return True

    def _on_training_end(self) -> None:
        if self.pbar:
            curr = int(self.model.num_timesteps)
            if curr > self.pbar.n:
                self.pbar.update(curr - self.pbar.n)
            self.pbar.close()


class EfficientEvalCallback(EvalCallback):
    """효율적인 평가 콜백 - 에이전트 경로만 시각화"""
    
    def __init__(self, eval_env, plot_dir="./eval_plots", **kwargs):
        super().__init__(eval_env=eval_env, **kwargs)
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # 시각화 설정
        sns.set_theme()
        plt.style.use('seaborn-v0_8')

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        
        # 평가가 실제로 수행되었는지 확인
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._create_path_visualization()
        
        return continue_training

    def _create_path_visualization(self):
        """에이전트 경로를 시각화한 정적 이미지 생성"""
        try:
            # 새 환경에서 에피소드 실행하여 경로 수집
            test_env = BattleSsafyEnv(size=16, render_mode=None)
            obs, _ = test_env.reset()
            
            # 경로 추적
            agent_path = []
            done, truncated = False, False
            max_steps = 200
            step_count = 0
            
            while not (done or truncated) and step_count < max_steps:
                # 현재 위치 저장
                agent_path.append(tuple(test_env._agent_pos))
                
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = test_env.step(action)
                step_count += 1
            
            # 마지막 위치 추가
            if step_count < max_steps:
                agent_path.append(tuple(test_env._agent_pos))
            
            # 시각화 생성
            self._plot_agent_path(test_env, agent_path, step_count, done)
            test_env.close()
            
        except Exception as e:
            print(f"[EfficientEvalCallback] 시각화 생성 중 오류: {e}")

    def _plot_agent_path(self, env, agent_path, steps, success):
        """에이전트 경로를 맵 위에 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 맵 + 경로 시각화
        self._plot_map_with_path(ax1, env, agent_path, success)
        
        # 2. 방문 빈도 히트맵
        self._plot_visit_heatmap(ax2, agent_path, env.H, env.W)
        
        # 제목과 정보
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        reward_info = f"Steps: {steps}, Success: {success}"
        fig.suptitle(f'Agent Evaluation - {self.num_timesteps} timesteps\n{reward_info}', 
                    fontsize=14, fontweight='bold')
        
        # 저장
        plot_path = os.path.join(self.plot_dir, f"eval_{self.num_timesteps}_{ts}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[EfficientEvalCallback] 평가 결과 저장: {plot_path}")

    def _plot_map_with_path(self, ax, env, agent_path, success):
        """맵과 에이전트 경로 시각화"""
        # 타일 색상 맵
        tile_colors = {
            G: 'lightgray',   # 평지
            R: 'darkgray',    # 바위
            W: 'lightblue',   # 물
            F: 'gold',        # 보급소
            X: 'red',         # 적 포탑
            E: 'blue',        # 적
            T: 'green',       # 나무
            S: 'wheat',       # 모래
            H: 'lightgreen',  # 우리 포탑
        }
        
        # 베이스 맵 시각화
        colored_map = np.zeros((env.H, env.W, 3))
        for y in range(env.H):
            for x in range(env.W):
                tile_type = env._base_map[y, x]
                color_name = tile_colors.get(tile_type, 'white')
                # matplotlib 색상을 RGB로 변환
                try:
                    color_rgb = plt.colors.to_rgb(color_name)
                    colored_map[y, x] = color_rgb
                except:
                    colored_map[y, x] = [1, 1, 1]  # 기본 흰색
        
        # 맵 표시 (y축 뒤집기 - 원점이 왼쪽 위)
        ax.imshow(colored_map, origin='upper')
        
        # 포탑 위치 표시
        for tx, ty in env._turrets:
            ax.plot(tx, ty, 'rX', markersize=12, markeredgewidth=3)
        
        for hx, hy in env._home_turrets:
            ax.plot(hx, hy, 'gX', markersize=12, markeredgewidth=3)
        
        # 에이전트 경로 표시
        if len(agent_path) > 1:
            path_x = [pos[0] for pos in agent_path]
            path_y = [pos[1] for pos in agent_path]
            
            # 경로 선
            ax.plot(path_x, path_y, 'orange', linewidth=3, alpha=0.8, label='Agent Path')
            
            # 시작점
            ax.plot(path_x[0], path_y[0], 'go', markersize=10, label='Start')
            
            # 끝점
            end_color = 'lime' if success else 'orange'
            ax.plot(path_x[-1], path_y[-1], 'o', color=end_color, markersize=10, label='End')
            
            # 방향 화살표 (몇 개만 표시)
            step_interval = max(1, len(agent_path) // 8)
            for i in range(0, len(agent_path)-1, step_interval):
                dx = path_x[i+1] - path_x[i]
                dy = path_y[i+1] - path_y[i]
                if dx != 0 or dy != 0:
                    ax.arrow(path_x[i], path_y[i], dx*0.3, dy*0.3, 
                            head_width=0.2, head_length=0.2, 
                            fc='darkorange', ec='darkorange', alpha=0.7)
        
        ax.set_xlim(-0.5, env.W-0.5)
        ax.set_ylim(-0.5, env.H-0.5)
        ax.set_title('Map with Agent Path')
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    def _plot_visit_heatmap(self, ax, agent_path, height, width):
        """방문 빈도 히트맵"""
        visit_count = np.zeros((height, width))
        
        # 방문 횟수 계산
        for x, y in agent_path:
            if 0 <= x < width and 0 <= y < height:
                visit_count[y, x] += 1
        
        # 히트맵 생성
        sns.heatmap(visit_count, annot=True, fmt='d', cmap='YlOrRd', 
                   ax=ax, cbar_kws={'label': 'Visit Count'})
        ax.set_title('Visit Frequency Heatmap')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')


class PolicyVisualizationCallback(BaseCallback):
    """정책 시각화 콜백 - 학습된 정책을 화살표로 표시"""
    
    def __init__(self, plot_dir="./policy_plots", save_freq=10000, verbose=1):
        super().__init__(verbose)
        self.plot_dir = plot_dir
        self.save_freq = save_freq
        os.makedirs(self.plot_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self._visualize_policy()
        return True

    def _visualize_policy(self):
        """학습된 정책을 시각화"""
        try:
            # 환경의 각 상태에서 최적 액션 계산
            test_env = BattleSsafyEnv(size=16, render_mode=None)
            
            # 각 위치에서의 최적 액션 계산
            action_map = np.zeros((test_env.H, test_env.W), dtype=int)
            value_map = np.zeros((test_env.H, test_env.W))
            
            for y in range(test_env.H):
                for x in range(test_env.W):
                    if test_env._is_passable((x, y)):
                        # 해당 위치에서의 관측값 생성
                        test_env._agent_pos = np.array([x, y])
                        obs = test_env._get_obs()
                        
                        # 정책으로부터 액션 확률 계산
                        action_probs = self.model.policy.get_distribution(obs).distribution.probs
                        best_action = int(np.argmax(action_probs.detach().cpu().numpy()))
                        max_prob = float(np.max(action_probs.detach().cpu().numpy()))
                        
                        action_map[y, x] = best_action
                        value_map[y, x] = max_prob
            
            # 시각화
            self._plot_policy_arrows(test_env, action_map, value_map)
            test_env.close()
            
        except Exception as e:
            if self.verbose:
                print(f"[PolicyVisualizationCallback] 오류: {e}")

    def _plot_policy_arrows(self, env, action_map, value_map):
        """정책을 화살표로 시각화"""
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # 액션을 화살표로 매핑
        action_arrows = {0: '→', 1: '↑', 2: '←', 3: '↓', 4: '💣', 5: '💥', 6: '🔓'}
        
        # 배경 맵 그리기 (간단한 버전)
        for y in range(env.H):
            for x in range(env.W):
                tile_type = env._base_map[y, x]
                
                # 타일에 따른 색상
                if tile_type == R:
                    ax.add_patch(plt.Rectangle((x-0.4, y-0.4), 0.8, 0.8, 
                                             facecolor='gray', alpha=0.5))
                elif tile_type == W:
                    ax.add_patch(plt.Rectangle((x-0.4, y-0.4), 0.8, 0.8, 
                                             facecolor='lightblue', alpha=0.5))
                elif tile_type == F:
                    ax.add_patch(plt.Rectangle((x-0.4, y-0.4), 0.8, 0.8, 
                                             facecolor='gold', alpha=0.5))
                
                # 통행 가능한 곳에만 화살표 표시
                if env._is_passable((x, y)):
                    action = action_map[y, x]
                    confidence = value_map[y, x]
                    
                    # 화살표 색상은 신뢰도에 따라
                    alpha = min(1.0, confidence * 2)
                    arrow = action_arrows.get(action, '?')
                    
                    ax.text(x, y, arrow, ha='center', va='center', 
                           fontsize=12, alpha=alpha, weight='bold')
        
        # 포탑 표시
        for tx, ty in env._turrets:
            ax.plot(tx, ty, 'rX', markersize=15, markeredgewidth=3)
        
        ax.set_xlim(-0.5, env.W-0.5)
        ax.set_ylim(-0.5, env.H-0.5)
        ax.set_title(f'Learned Policy - {self.num_timesteps} timesteps')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 저장
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.plot_dir, f"policy_{self.num_timesteps}_{ts}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"[PolicyVisualizationCallback] 정책 시각화 저장: {plot_path}")