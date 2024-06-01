# %%
import time
import numpy as np
from numba import njit
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox


# %%
def set_bbox_inches_tight(fig, ratio_margin=0.01):
    assert 0 <= ratio_margin <= 1
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    width, height = bbox.width, bbox.height # inches

    bb = [ax.get_tightbbox(fig.canvas.get_renderer()) for ax in fig.axes]
    tight_bbox_raw = Bbox.union(bb)
    tight_bbox = fig.transFigure.inverted().transform_bbox(tight_bbox_raw)

    l, b, w, h = tight_bbox.bounds # 0~1 value (relative)
    margin = ratio_margin * max(width, height)
    w_m = margin / width
    h_m = margin / height
    l -= w_m
    b -= h_m
    w += w_m * 2
    h += h_m * 2
    for ax in fig.axes:
        l2, b2, w2, h2 = ax.get_position().bounds
        r2, t2 = l2+w2, b2+h2
        l3 = (l2 - l) / w
        r3 = (r2 - l) / w
        b3 = (b2 - b) / h
        t3 = (t2 - b) / h
        w3 = r3 - l3
        h3 = t3 - b3
        ax.set_position((l3, b3, w3, h3))

    fig.set_size_inches((width+2*margin, height+2*margin))


# %%
radius_ = .5 * np.array((1, 1.33, 2, 2.25, 2.67, 3.5, 4, 5, 5.67, 6.75, 8))
color_ = np.array((
    (228, 8, 8),
    (251, 109, 75),
    (165, 106, 255),
    (255, 186, 2),
    (252, 134, 25),
    (244, 21, 20),
    (254, 240, 120),
    (255, 183, 175),
    (248, 232, 7),
    (141, 195, 17),
    ( 12, 99, 11),
))/255

@njit(fastmath=True, cache=True)
def set_ball_mrI(ball):
    ball_type = ball['type']
    ball['m'] = 1
    ball['1/m'] = ball['m']
    ball['r'] = radius_[ball_type]
    ball['I'] = .5 * ball['m'] * ball['r']**2
    ball['1/I'] = 1 /ball['I']

    
# Constants
# dt = 1/30 
dt = 1/60 
inv_dt = 1/dt
n_substep = 10
inv_n_substep = 1/n_substep

gamma_v = .9 ** (dt * inv_n_substep)
gamma_omega = .2 ** (dt * inv_n_substep)

elasticity_collision = 1 # max 2 --> complete elastic
elasticity_overlap = .2 # bais elastic constant

coef_friction = .01

x_box_min = 0
x_box_max = 14.25
y_box_min = 0
y_box_max = 17
cursor_y = y_box_max + .05
max_n_ball = np.ceil(x_box_max * y_box_max).astype(int)

@njit(fastmath=True, cache=True)
def get_contact_(ball_, n_ball, contact_):
    should_combine = False
    i_ball_combine = -1
    i_ball_2_combine = -1

    n_contact = 0
    for i_ball in range(n_ball-1):
        ball = ball_[i_ball]
        x = ball['x']
        y = ball['y']
        r = ball['r']

        for i_ball_2 in range(i_ball+1, n_ball):
            ball_2 = ball_[i_ball_2]
            x2 = ball_2['x']
            y2 = ball_2['y']
            r2 = ball_2['r']
            
            if (x-x2)**2 + (y-y2)**2 <= (r+r2)**2:
                contact_[n_contact]['i_ball'] = i_ball
                contact_[n_contact]['i_ball_2'] = i_ball_2
                n_contact += 1

                if should_combine == False:
                    if ball['type'] == ball_2['type']:
                        should_combine = True
                        i_ball_combine = i_ball
                        i_ball_2_combine = i_ball_2

    return contact_, n_contact, should_combine, i_ball_combine, i_ball_2_combine

@njit(fastmath=True, cache=True)
def combine_ball(ball_, n_ball, should_combine, i_ball, i_ball_2):
    diff_score = 0
    if should_combine is True:
        ball = ball_[i_ball]
        ball_2 = ball_[i_ball_2]
        ball_type = ball['type']
        if ball_type < 10:
            x = ball['x']
            y = ball['y']
            x2 = ball_2['x']
            y2 = ball_2['y']
            r = radius_[ball['type']+1]
            x = max(x_box_min+r, .5*(x+x2))
            x = min(x_box_max-r, x)
            y = .5*(y + y2)
            ball['x'] = x
            ball['y'] = y
            ball['vx'] = 0
            ball['vy'] = 0
            ball['theta'] = 0
            ball['omega'] = 0
            ball['Px'] = 0
            ball['Py'] = 0
            ball['L'] = 0
            ball['type'] += 1
            set_ball_mrI(ball)

            ball_[i_ball_2] = ball_[n_ball -1]
            n_ball -= 1
            
        else:
            ball_[i_ball_2] = ball_[n_ball -1]
            n_ball -= 1

            ball_[i_ball] = ball_[n_ball -1]
            n_ball -= 1

        diff_score = ((ball_type + 1) * (ball_type + 2)) // 2

    return n_ball, diff_score

@njit(fastmath=True, cache=True)
def system_ball_physics_numba(
    ball_,
    n_ball,
    contact_,
    dt,
    n_substep,
):
    dt = dt * inv_n_substep
    diff_score_total = 0
    for _ in range(n_substep):
        contact_, n_contact, should_combine, i_ball_combine, i_ball_2_combine = get_contact_(ball_, n_ball, contact_)
        ball_['Px'][:] = 0
        ball_['Py'][:] = 0
        ball_['L'][:] = 0

        for i_contact in range(n_contact):
            i_ball = contact_[i_contact]['i_ball']
            i_ball_2 = contact_[i_contact]['i_ball_2']
            ball = ball_[i_ball]
            ball_2 = ball_[i_ball_2]
            x = ball['x']
            y = ball['y']
            vx = ball['vx']
            vy = ball['vy']
            omega = ball['omega']
            r = ball['r']
            inv_m = ball['1/m']
            inv_I = ball['1/I']

            x2 = ball_2['x']
            y2 = ball_2['y']
            vx2 = ball_2['vx']
            vy2 = ball_2['vy']
            omega2 = ball['omega']
            r2 = ball_2['r']
            inv_m2 = ball_2['1/m']
            inv_I2 = ball_2['1/I']

            dx = x2 - x
            dy = y2 - y
            d = np.sqrt(dx**2 + dy**2)
            inv_d = 1/d
            cos = dx*inv_d # direction of r, -r2, n
            sin = dy*inv_d

            dvx = vx2 + omega2 * r2 * (--sin) - vx - omega * r * (-sin)
            dvy = vy2 + omega2 * r2 * (-cos) - vy - omega * r * cos
            
            v_n = dvx * cos + dvy * sin  # _n for normal, _t for tangential
            m_n = 1 / (inv_m + inv_m2)  # effective mass for normal impact
            P_n = max(-elasticity_collision*m_n*v_n + elasticity_overlap*max(r +r2 -d, 0)*inv_dt, 0) # normal impact
            Px_n = P_n * cos
            Py_n = P_n * sin
            ball['Px'] -= Px_n
            ball['Py'] -= Py_n
            ball_2['Px'] += Px_n
            ball_2['Py'] += Py_n

            v_t = dvx * sin + dvy * -cos
            m_t = 1 / (inv_m + inv_m2 + inv_I * r**2 + inv_I2 * r2**2)
            P_t = -coef_friction*m_t*v_t
            Px_t = P_t * sin
            Py_t = P_t * -cos
            ball['Px'] += Px_t
            ball['Py'] += Py_t
            ball_2['Px'] -= Px_t
            ball_2['Py'] -= Py_t
            ball['L'] += r*P_t
            ball_2['L'] += r2*P_t

            
        # Update and constrain
        for i_ball in range(n_ball):
            ball = ball_[i_ball]
            x = ball['x']
            y = ball['y']
            vx = ball['vx']
            vy = ball['vy']
            omega = ball['omega']
            r = ball['r']
            m = ball['m']
            inv_m = ball['1/m']
            inv_I = ball['1/I']

            if ball['x'] <= x_box_min +ball['r']:
                cos = -1
                sin = 0
                dvx = - vx - omega * r * (-sin)
                dvy = - vy - omega * r * cos
                v_n = dvx * cos + dvy * sin
                m_n = m
                P_n = max(-elasticity_collision*m_n*v_n + elasticity_overlap*max(x-r, 0)/dt, 0)
                Px_n = P_n * cos
                Py_n = P_n * sin
                ball['Px'] -= Px_n
                ball['Py'] -= Py_n

                v_t = dvx * sin + dvy * -cos
                m_t = 1 / (inv_m + inv_I * r**2) 
                P_t = -elasticity_collision*m_t*v_t
                Px_t = P_t * sin
                Py_t = P_t * -cos
                ball['Px'] -= Px_t
                ball['Py'] -= Py_t
                ball['L'] += r*P_t

            if ball['x'] >= x_box_max -ball['r']:
                cos = 1
                sin = 0
                dvx = - vx - omega * r * (-sin)
                dvy = - vy - omega * r * cos
                v_n = dvx * cos + dvy * sin
                m_n = m
                P_n = max(-elasticity_collision*m_n*v_n + elasticity_overlap*max(x+r-x_box_max, 0)/dt, 0)
                Px_n = P_n * cos
                Py_n = P_n * sin
                ball['Px'] -= Px_n
                ball['Py'] -= Py_n

                v_t = dvx * sin + dvy * -cos
                m_t = 1 / (inv_m + inv_I * r**2) 
                P_t = -elasticity_collision*m_t*v_t
                Px_t = P_t * sin
                Py_t = P_t * -cos
                ball['Px'] -= Px_t
                ball['Py'] -= Py_t
                ball['L'] += r*P_t

            if ball['y'] <= y_box_min +ball['r']:
                cos = 0
                sin = -1
                dvx = - vx - omega * r * (-sin)
                dvy = - vy - omega * r * cos
                ball['vy'] = 0

                v_t = dvx * sin + dvy * -cos
                m_t = 1 / (inv_m + inv_I * r**2) 
                P_t = -elasticity_collision*m_t*v_t
                Px_t = P_t * sin
                ball['Px'] -= Px_t
                ball['L'] += r*P_t
            else:
                ball['Py'] -= dt * 25 * ball['m']

            inv_m = ball['1/m']
            ball['vx'] = gamma_v*ball['vx'] + ball['Px'] * inv_m
            ball['vy'] = gamma_v*ball['vy'] + ball['Py'] * inv_m
            ball['omega'] = gamma_omega*ball['omega'] + min(max(ball['L'], -1), 1) * ball['1/I']

            ball['x'] = max(x_box_min +ball['r'], ball['x'] +dt*ball['vx'])
            ball['x'] = min(x_box_max -ball['r'], ball['x'])
            ball['y'] = max(y_box_min +ball['r'], ball['y'] +dt*ball['vy'])
            ball['theta'] = ball['theta'] +ball['omega']*dt
            ball['theta'] -= (ball['theta'] // (2*np.pi))*(2*np.pi)
        n_ball, diff_score = combine_ball(ball_, n_ball, should_combine, i_ball_combine, i_ball_2_combine)
        diff_score_total += diff_score

    return n_ball, diff_score_total

class World:
    def __init__(self):
        # Component (ECS)
        self.ball_ = np.zeros(
            max_n_ball,
            dtype=[
                # state
                ('x', np.float64),
                ('y', np.float64),
                ('vx', np.float64),
                ('vy', np.float64),
                
                # body properties
                ('type', np.int32), # 0~10
                ('theta', np.float64),
                ('omega', np.float64),
                ('r', np.float64),
                ('m', np.float64),
                ('1/m', np.float64),
                ('I', np.float64),
                ('1/I', np.float64),
                
                # applied forces
                ('Fx', np.float64),
                ('Fy', np.float64),
                ('tau', np.float64),
                ('Px', np.float64),
                ('Py', np.float64),
                ('L', np.float64),
            ]
        )


        # self.n_contact = 0
        self.contact_ = np.zeros(
            2*max_n_ball,
            dtype=[
                # state
                ('i_ball', np.int32),
                ('i_ball_2', np.int32),
            ]
        )
        self.i_ball_not_available = np.zeros(max_n_ball, dtype=np.int32)


    def run(self):
        # Variables
        self.score = 0
        self.is_down = False
        self.add_block = 0
        self.is_not_closed = True
        self.cursor_x = x_box_max * .5
        self.n_ball = 0
        self.n_ball_prev = 0
        self.ball_[:] = 0
        self.next_ball_type_ = np.random.randint(5, size=2)
        # self.next_ball_type_ = np.array([9, 9, 9, 9, 9, 6, 6, 3, 2, 1], dtype=int)
        
        self.fig, self.ax = plt.subplots(dpi=150)
        self.ax.set_xlim((x_box_min-.1, x_box_max+.1))
        self.ax.set_ylim((y_box_min-.1, y_box_max+.1))
        self.ax.set_aspect(1)
        set_bbox_inches_tight(self.fig)
        
        self.cursor = mpatches.Circle((0, 0), 0, ec="none", animated=True)
        self.ax.add_artist(self.cursor)
        self.cursor.set_radius(radius_[self.next_ball_type_[0]])
        self.cursor.set_color(color_[self.next_ball_type_[0]])

        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.fig.canvas.mpl_connect("draw_event", self.on_draw)
        self.artist_ball_ = []
        self.artist_line_ball_ = []
        for i in range(max_n_ball):
            artist = mpatches.Circle((0, 0), 0, ec="none", animated=True)
            self.ax.add_artist(artist)
            artist.set_visible(False)
            self.artist_ball_.append(artist)
            artist_line, = self.ax.plot([], [], color=(1, 1, 1), linewidth=2, animated=True)
            self.artist_line_ball_.append(artist_line)
        margin = .05 * min(x_box_max, y_box_max)
        self.text_score = self.ax.text(margin, y_box_max -margin, '', va='top', ha='left', animated=True)

        plt.ion()
        plt.show()

        self.fig.canvas.draw()

        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)


        t_0 = time.time()
        while self.is_not_closed == True:
            self.n_ball_prev = self.n_ball
            self.system_ball_add()
            self.system_ball_physics()
            self.system_display()

            self.fig.canvas.flush_events() # Refresh Input Buffer

            t = time.time() # System Time
            if (t-t_0) < dt:
                time.sleep(dt - (t-t_0))
                print(f"sleeped : {dt - (t-t_0):.5f} sec")
            t_0 = t
        self.is_not_closed = True
        plt.close(self.fig)

    def system_ball_add(self):
        self.add_block = max(0, self.add_block-1)
        if self.is_down == True:
            if self.n_ball < max_n_ball:
                if self.add_block <= 0:
                    self.add_block = round(0.5/dt)
                    i_ball = self.n_ball
                    self.n_ball += 1
                    ball = self.ball_[i_ball]
                    ball['x'] = self.cursor_x
                    ball['y'] = cursor_y
                    ball['vx'] = 0
                    ball['vy'] = 0
                    ball['theta'] = 0
                    ball['omega'] = 0
                    ball['Px'] = 0
                    ball['Py'] = 0
                    ball['L'] = 0
                    ball['type'] = self.next_ball_type_[0]
                    set_ball_mrI(ball)
                    
                    self.cursor.set_radius(radius_[self.next_ball_type_[1]])
                    self.cursor.set_color(color_[self.next_ball_type_[1]])
                    self.next_ball_type_[:-1] = self.next_ball_type_[1:]
                    self.next_ball_type_[-1] = np.random.randint(5)

    def system_ball_physics(self):
        self.n_ball, diff_score = system_ball_physics_numba(
            self.ball_,
            self.n_ball,
            self.contact_,
            dt,
            n_substep,
        )
        self.score += diff_score
        
    def draw_artist_animated(self):
        for artist in [self.cursor] +self.artist_ball_ +self.artist_line_ball_:
            self.fig.draw_artist(artist)

    def system_display(self):
        self.fig.canvas.restore_region(self.bg)
        self.cursor.set_center((self.cursor_x, cursor_y))

        # balls
        for i_ball in range(self.n_ball):
            ball = self.ball_[i_ball]
            x = ball['x']
            y = ball['y']
            r = ball['r']
            theta = ball['theta']

            artist = self.artist_ball_[i_ball]
            artist.set_radius(r)
            artist.set_color(color_[ball['type']])
            artist.set_center((x, y))
            artist.set_visible(True)


            artist_line = self.artist_line_ball_[i_ball]
            artist_line.set_data([x, x+r*np.cos(theta)], [y, y+r*np.sin(theta)])
            artist_line.set_visible(True)

        for i_ball in range(self.n_ball, self.n_ball_prev):
            self.artist_ball_[i_ball].set_visible(False)
            self.artist_line_ball_[i_ball].set_visible(False)

        # score
        self.text_score.set_text(f"Score: {self.score}")
        self.ax.draw_artist(self.text_score)

        self.draw_artist_animated()
        self.fig.canvas.blit(self.fig.bbox)


    # Event Handler
    def on_move(self, event):
        x = event.xdata
        y = event.ydata
        if (x is not None) and (y is not None):
            self.cursor_x = np.clip(
                x,
                x_box_min +radius_[self.next_ball_type_[0]],
                x_box_max -radius_[self.next_ball_type_[0]],
            )
            
    def on_press(self, event):
        if event.button == 1:
            self.is_down = True

    def on_release(self, event):
        if event.button == 1:
            self.is_down = False

    def on_close(self, event):
        self.is_not_closed = False
        print('Closed Game')

    def on_draw(self, event):
        if event is not None:
            if event.canvas != self.fig.canvas:
                raise RuntimeError
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.draw_artist_animated()


# %%
try:
    world = World()
    world.run()
except KeyboardInterrupt:
    print("Stopping Game.")
