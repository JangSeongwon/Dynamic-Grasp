# Basic modules
# 4개의 함수 기본

# 동일 class에서 obs를 정의하여 여기에 넣어 보냄

""" Into class PandaEnv(MujocoEnv) """
def render_obs(self, mode=None, width=448, height=448, camera_id=None):
    self._render_callback()
    data = []
    for cam in self.cameras:
        if cam == 'first_person':
            if width != 84:
                old_w = old_h = width  # 84
            else:
                old_w = old_h = 448  # 84
            width1 = 640
            height1 = 640
            img = self.sim.render(width1, height1, camera_name=cam, depth=False)[::-1, :, :]
            img = img[(width1 - int(old_w)):, (width1 - int(old_w)):]  # 84x84
            img[:old_w - int(old_w * 3 / 4), :] = np.zeros((old_w - int(old_w * 3 / 4), old_w, 3))
            if width == 84:
                img = cv2.resize(img, dsize=(84, 84), interpolation=cv2.INTER_CUBIC)
        else:
            img = self.sim.render(width, height, camera_name=cam, depth=False)[::-1, :, :]
        data.append(img)    # 빈 행렬에 image를 벡터안에 추가
    return np.asarray(data)   # asarray : 복사본없이 생성

def render(self, mode='human', width=500, height=500, depth=False, camera_id=0):
    return super(BaseEnv, self).render(mode, width, height)


def _get_image_obs(self):
    return self.render_obs(mode='rgb_array', width=self.image_size, height=self.image_size)

def _render_callback(self):
    self.sim.forward()
    
