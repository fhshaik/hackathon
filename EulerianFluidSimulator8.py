import tkinter as tk
import numpy as np
from PIL import Image, ImageTk, ImageDraw


class Simulation:
    def __init__(self, root):
        self.root = root
        self.create_widgets()

    def create_widgets(self):
        self.canvas = tk.Canvas(self.root, width=400, height=400, bg='white')
        self.canvas.pack()

        self.reset_button = tk.Button(self.root, text="Reset", command=self.reset_simulation)
        self.reset_button.pack(side="left")

        self.image = Image.new("RGB", (100, 100), "white")
        self.photo = ImageTk.PhotoImage(self.image)
        
        self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.quit_button = tk.Button(self.root, text="QUIT", fg="red", command=self.root.destroy)
        self.quit_button.pack(side="right")

        self.message = "Red Blood Cells in Blood Plasma Eulerian Fluid Simulator"
        self.message_label = tk.Label(self.root, text=self.message)
        self.message_label.pack()

        #variables
        self.N = 100
        self.density = np.zeros((self.N, self.N)).astype(float)
        self.density0 = np.zeros((self.N, self.N)).astype(float)
        self.u = np.zeros((self.N, self.N)).astype(float)
        self.v = np.zeros((self.N, self.N)).astype(float)
        self.u0 = np.zeros((self.N, self.N)).astype(float)
        self.v0 = np.zeros((self.N, self.N)).astype(float)
        self.num_particles = 20
        self.dt = 0.0001

        self.decay=0.001
        self.charge =1000
        self.wall_charge=100
        self.centering_force=10
        self.viscosity =0.01
        self.diff=0.01
        self.epsilon =0.9
        self.max_distance=10

        self.particles = np.random.rand(self.num_particles, 4)*96 + 2
        self.particles[:, 2:] =0.0
        self.is_mouse_pressed = False

        #binding events
        self.canvas.bind("<Button-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

        #simulation
        self.update_simulation()

    def reset_simulation(self):
        self.density = np.zeros((self.N, self.N)).astype(float)
        self.density0 = np.zeros((self.N, self.N)).astype(float)
        self.u = np.zeros((self.N, self.N)).astype(float)
        self.v = np.zeros((self.N, self.N)).astype(float)
        self.u0 = np.zeros((self.N, self.N)).astype(float)
        self.v0 = np.zeros((self.N, self.N)).astype(float)
        self.particles = np.random.rand(self.num_particles, 4)*96 + 2
        self.particles[:, 2:] =0.0
        self.message_label.config(text="Simulation reset")

    def on_mouse_press(self, event):
        self.is_mouse_pressed = True

    def on_mouse_drag(self, event):
        if self.is_mouse_pressed:
            x, y = event.x // 4, event.y // 4
            if 0 <= x < self.N and 0 <= y < self.N:
                self.density[y][x] += 10
                dx = (event.x - self.prev_mouse_x) * 0.0001
                dy = (event.y - self.prev_mouse_y) * 0.0001
                self.u += dx
                self.v += dy
            self.prev_mouse_x, self.prev_mouse_y = event.x, event.y

    def on_mouse_release(self, event):
        self.is_mouse_pressed = False

    def update_simulation(self):
        if self.is_mouse_pressed:
            x, y = self.root.winfo_pointerx() - self.root.winfo_rootx(), self.root.winfo_pointery() - self.root.winfo_rooty()
            x, y = x//4, y//4
            if 0 <= x < self.N and 0 <= y < self.N:
                self.density[y][x] += 10
                dx = (self.root.winfo_pointerx() - self.prev_mouse_x) * 0.0001
                dy = (self.root.winfo_pointery() - self.prev_mouse_y) * 0.0001
                self.u += dx
                self.v += dy
            
        self.prev_mouse_x = self.root.winfo_pointerx()
        self.prev_mouse_y = self.root.winfo_pointery()
        self.denStep(self.N, self.density, self.density0, self.u, self.v, self.diff, self.dt)
        self.velStep(self.N, self.u, self.v, self.u0, self.v0, self.viscosity, self.dt)
        self.update_particles(self.particles, self.dt, self.N, self.u, self.v, 4, self.decay, self.charge,self.wall_charge,self.centering_force,self.epsilon,self.max_distance)
        self.render_dens_particles(self.particles)
        self.root.after(1, self.update_simulation)

    def update_particles(self, particles, dt, N, u, v, boundary, decay,charge,wall_charge,centering_force, epsilon,max_distance):
        #im assuming this helped me from reaching resonance 
        dt0 = dt*np.power(N,epsilon)

        updated_particles = []
        for particle in particles:
            x_index = particle[0]
            y_index = particle[1]

            #boundary conditions for variables
            i0 = np.clip(x_index, 10, N - 10).astype(int)
            i1 = np.clip(i0 + 1, 10, N - 10).astype(int)
            j0 = np.clip(y_index, 10, N - 10).astype(int)
            j1 = np.clip(j0 + 1, 10, N - 10).astype(int)


            #we use a binlinear interpolation algorithm as per Jos Stam's paper
            s1 = x_index - i0
            s0 = 1 - s1
            t1 = y_index - j0
            t0 = 1 - t1

            interpolated_u = s0 * (t0 * u[i0, j0] + t1 * u[i0, j1]) + s1 * (t0 * u[i1, j0] + t1 * u[i1, j1])
            interpolated_v = s0 * (t0 * v[i0, j0] + t1 * v[i0, j1]) + s1 * (t0 * v[i1, j0] + t1 * v[i1, j1])
            
            new_u = interpolated_u +decay*particle[2]+ self.calculate_force(particle,particles,(N,N),charge,wall_charge,centering_force)[0]*dt0
            new_v = interpolated_v + decay*particle[3] + self.calculate_force(particle,particles,(N,N),charge,wall_charge,centering_force)[1]*dt0
            max_speed = 100

            new_u = np.clip(new_u, -max_speed, max_speed)
            new_v = np.clip(new_v, -max_speed, max_speed)
            new_x = particle[0] + new_u*dt0
            new_y = particle[1] + new_v*dt0


            #boundary conditions for particles
            if new_x < boundary:
                new_x = boundary +(boundary -new_x)
                new_u = -new_u
            if new_x> N - boundary:
                new_x=N -boundary -(new_x -(N -boundary))
                new_u = -new_u
            if new_y < boundary:
                new_y = boundary+(boundary - new_y)
                new_v = -new_v
            if new_y > N - boundary:
                new_y =N -boundary -(new_y - (N -boundary))
                new_v = -new_v
            
            #calculates collissions
            for other_particle in particles:
                if np.array_equal(particle, other_particle):
                    continue  
                distance = np.linalg.norm(particle[:2] - other_particle[:2])
                if distance < max_distance:
                    direction = (particle[:2] - other_particle[:2]) / distance
                    displacement = (max_distance - distance) * direction
                    #basically every action has an opposite equal reaction
                    particle[:2] += displacement / 2
                    other_particle[:2] -= displacement / 2
            
            updated_particles.append([new_x, new_y,new_u,new_v])

        particles[:] = updated_particles

        return particles
    
    def calculate_force(self, particle, particles, container_size,charge,wall_charge,centering_force):
        force = np.zeros(2)


        # particle-particle interactions
        for other_particle in particles:
            if np.array_equal(particle[:2], other_particle[:2]):
                continue
            dist = np.linalg.norm(np.array(particle[:2]) - np.array(other_particle[:2]))
            if dist == 0:
                random_direction = np.random.rand(2) - 0.5  
                force += random_direction * charge*charge

            force_magnitude = charge / (dist ** 2)
            force_direction = (np.array(other_particle[:2]) - np.array(particle[:2])) / dist
            force += (force_magnitude * force_direction)

        dist_x_min = particle[0]
        dist_x_max = container_size[0] - particle[0] 
        force_x_min = wall_charge / (dist_x_min ** 2)
        force_x_max = wall_charge / (dist_x_max ** 2)
        force[0] += force_x_min - force_x_max  


        dist_y_min = particle[1]  
        dist_y_max = container_size[1] - particle[1]  
        force_y_min = wall_charge/(dist_y_min ** 2)
        force_y_max = wall_charge/(dist_y_max ** 2)
        force[1] += force_y_min-force_y_max  

        center_x = container_size[0]/2
        center_y = container_size[1]/2
        centering_direction = np.array([center_x, center_y]) - np.array(particle[:2])
        force += centering_force * centering_direction

        max_force_magnitude = 1000000
        force_magnitude = np.linalg.norm(force)
        if force_magnitude > max_force_magnitude:
            force = force * (max_force_magnitude / force_magnitude)

        return force

    def apply_no_slip_boundary(self, x, toggle):
        if(toggle==1):
            x[:, :2] = 0  
            x[:, -2:] = 0  
            x[:2, :] = 0
            x[-2:, :] = 0 
        else:
            x[0, :] = x[1, :]
            x[-1, :] = x[-2, :] 
            x[:, 0] = x[:, 1] 
            x[:, -1] = x[:, -2] 

    def render_dens_particles(self, particle_array):
        density_image = self.render_density()
        
        draw = ImageDraw.Draw(density_image)
        for particle in particle_array:
            
            if 0<particle[0]<self.N and 0<particle[1]<self.N:
                x, y = int(particle[0]), int(particle[1])
                draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill="red")
        density_image = density_image.resize((400, 400))
        self.photo = ImageTk.PhotoImage(density_image)
        
        self.canvas.itemconfig(self.image_id, image=self.photo)

        
    def denStep(self, N, x, x0, u, v, diff, dt):

        #i was originally confused by the swaps, but its an efficient way to reuse memory in Joe Stam's paper
        x, x0 = self.swap(x, x0)
        #spreads things around
        self.diffusion(N, 0, x, x0, diff, dt,1)
        x, x0 = self.swap(x, x0)
        #uses velocity field to move things around
        self.advect(N, 0, x, x0, u, v, dt)

    #originally this was in a 3d for loop but it ran so slowly
    #luckily I discovered that python is extremely good at matrix type operations
    #the for k in range(10) is just a linear approximation method to find solution of equations called gauss seidel method
    def diffusion(self, N, b, x, x0, diff, dt,toggle):
        a = dt * diff * N * N
        for k in range(10):
            x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a * (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:])) / (1 + 4 * a) 
            
        #applying boundary conditions
        self.apply_no_slip_boundary(x,toggle)

    def swap(self, x, y):
        return y, x
    
    #The advect operation in stable fluid algorithm uses something called backwards interpolation
    #You take your velocity go backwards and get the velocity at the backwards point and transpose it to current position
    #The purpose is because it helps make the algorithm stable. ALso, it makes sense, fluids trace out a path

    def advect(self, N, b, d, d0, u, v, dt):
        dt0 = dt*N
        i, j = np.meshgrid(np.arange(1, N-1), np.arange(1, N-1), indexing='ij')

        #indexes
        x = i - dt0 * u[i, j]
        y = j - dt0 * v[i, j]

        x = np.clip(x, 0.5, N - 0.5)
        y = np.clip(y, 0.5, N - 0.5)

        #indexes for backwards interp
        i0 = np.floor(x).astype(int)
        i1 = i0 + 1
        j0 = np.floor(y).astype(int)
        j1 = j0 + 1

        #boundary conditions for indexes
        i0 = np.clip(i0, 0, N - 1)
        i1 = np.clip(i1, 0, N - 1)
        j0 = np.clip(j0, 0, N - 1)
        j1 = np.clip(j1, 0, N - 1)

        #backwards interpolation
        s1 = x - i0
        s0 = 1 - s1
        t1 = y - j0
        t0 = 1 - t1
        d[1:N-1, 1:N-1] = (s0 *(t0 * d0[i0, j0]) + t1 *(s0 * d0[i1, j0]) + s1 * (t0 * d0[i0, j1]) + t1 * d0[i1, j1])

    
    def normalize_density(self,density):
        if np.unique(density).size == 1:
            return density.astype(np.uint8)


        min_val = np.min(density)
        max_val = np.max(density)

        return (density - min_val) / (max_val - min_val)
    

    def render_density(self):
        #I take the normalized density and add a root operation to make lower values rise closer to 1
        density_normalized = np.power(self.normalize_density(self.density),0.05)
        
        #I set up the exact colors in order to get pink
        density_scaled_r = (np.power(density_normalized,0.5) * 255).astype(np.uint8)
        density_scaled_g = (np.power(density_normalized,0.5) * 180).astype(np.uint8)
        density_scaled_b = (np.power(density_normalized,0.5) * 203).astype(np.uint8)

        #I think this has to do with transparency
        alpha_mask = (density_normalized * 255).astype(np.uint8)
        
        
        density_rgba = np.stack((density_scaled_r, density_scaled_g, density_scaled_b, alpha_mask), axis=-1)
        
        #creates image
        density_image = Image.fromarray(density_rgba, mode="RGBA")

        return density_image
        
        #updates canvas
        self.photo = ImageTk.PhotoImage(density_image)
        self.canvas.itemconfig(self.image_id, image=self.photo)

    #second important step of algorithm
    def velStep(self,N,u,v,u0,v0,visc,dt):
        #many parts are same as a Density Step,so I will not explain, but the project operation is new
        #It comes from the Helmholtz Decomposition Theorem, the fact we can decompose a vector field into divergence and divergence free parts
        #projection operation just gets the divergence free part. Why?
        #we want 0 divergence as the Eulerian Fluid simulator is a numerical approximation to the Navier Stokes Equations
        #0 divergence for velocity is a must for mass continuity
        u,u0 = self.swap(u,u0)
        self.diffusion(N,1,u,u0,visc,dt,1)
        v,v0 = self.swap(v,v0)
        self.diffusion(N,2,v,v0,visc,dt,1)
        self.project(N,u,v,u0,v0)
        u,u0 = self.swap(u,u0)
        v,v0 = self.swap(v,v0)
        self.advect( N, 1, u, u0, u0, v0, dt)
        self.advect( N, 2, v, v0, u0, v0, dt)
        self.project(N,u,v,u0,v0)
        max_velocity = 100.0  #just here to help not get unphysical results
        u[u>max_velocity] = max_velocity
        u[u<-max_velocity] = -max_velocity
        v[v>max_velocity] = max_velocity
        v[v<-max_velocity] = -max_velocity


    def project(self, N, u, v, p, div):

        h = 1/N
        #I assume knowledge of the divergence formula
        div[1:-1, 1:-1] = -0.5*h*(u[2:, 1:-1]-u[:-2, 1:-1]+v[1:-1, 2:]-v[1:-1, :-2])
        p[:] = 0 


        N = u.shape[0]
        M = 2 #border

        #this is basically something that pushes fluids back from leaving the box, important
        u[:M, :] = np.abs(u[:M, :])
        u[-M:, :] = np.abs(u[-M:, :])
        v[:M, :] = -np.abs(v[:M, :])
        v[-M:, :] = -np.abs(v[-M:, :])

        u[:, :M] = -np.abs(u[:, :M])
        u[:, -M:] = -np.abs(u[:, -M:])
        v[:, :M] = np.abs(v[:, :M])
        v[:, -M:] = np.abs(v[:, -M:])

        #whenever there's a random loop like this it's gauss seidel method
        for k in range(20):
            p[1:-1, 1:-1] = (div[1:-1, 1:-1] + p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4
        
        u[1:-1, 1:-1] -= 0.5*(p[2:, 1:-1] - p[:-2, 1:-1])/h
        v[1:-1, 1:-1] -= 0.5*(p[1:-1, 2:] - p[1:-1, :-2])/h
            
    
    def on_mouse_press(self, event):
        self.is_mouse_pressed = True

    def on_mouse_drag(self, event):
        pass

    def on_mouse_release(self, event):
        self.is_mouse_pressed = False

root = tk.Tk()
root.configure(bg='white')
sim = Simulation(root)
root.mainloop()
