import numpy as np
from numpy import pi  

# def la_velocita(deg,time,v_time,vec_time,coordinates,v_coo,vec_coo,vec_vel,velocity,vec_speed,vec_angle) :
def la_velocita(deg,time,v_coo,vec_vel,velocity,vec_speed,vec_angle) :

    if deg == 1:
        #Backward difference    
        velocity = v_coo[1,:]-v_coo[0,:]
        speed = np.sqrt(velocity.dot(velocity)) 
        try:
                tg = velocity[1]/velocity[0]
                angle = np.arctan(tg)

        except:
            angle = np.pi/2

        vec_vel = np.vstack([vec_vel, velocity])
        vec_speed = np.vstack([vec_speed, speed])
        vec_angle = np.vstack([vec_angle, angle])

    elif deg == 2:
        #Centered difference (Parabolic)
        # it calculates velocity for n-1 frames!!!
        if time == 1:
            pass
        else:
            velocity = 0.5*(v_coo[2,:]-v_coo[0,:])
            speed = np.sqrt(velocity.dot(velocity))
            try:
                tg = velocity[1]/velocity[0]
                angle = np.arctan(tg)

            except:
                angle = np.pi/2

            vec_vel = np.vstack([vec_vel, velocity])
            vec_speed = np.vstack([vec_speed, speed])
            vec_angle = np.vstack([vec_angle, angle])

    else:
        print("Incorrect velocity degree, deg = 1, 2")

    return angle, speed, velocity, vec_vel, vec_angle, vec_speed