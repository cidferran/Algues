import numpy as np
from numpy import pi  

# def la_velocita(deg,time,v_time,vec_time,coordinates,v_coo,vec_coo,vec_vel,velocity,vec_speed,vec_angle) :
def la_velocita(deg,time,v_coo,vec_vel,velocity,vec_speed,vec_angle) :

    v = velocity
    # v_coo = np.vstack([v_coo,coordinates])
    # v_coo = np.delete(v_coo,0,0)  
    # vec_coo = np.vstack([vec_coo,coordinates])

    # v_time = np.vstack([v_time,time])
    # v_time = v_time[deg+1-deg:deg+1,:]
    # vec_time= np.vstack([vec_time,time])

    if deg == 1:
        #Backward difference    
        v= v_coo[1,:]-v_coo[0,:]
        speed = np.sqrt(v.dot(v)) 
        try:
                tg = v[1]/v[0]
                angle = np.arctan(tg)

        except:
            angle = pi/2

        vec_vel = np.vstack([vec_vel, v])
        vec_speed = np.vstack([vec_speed, speed])
        vec_angle = np.vstack([vec_angle, angle])

    elif deg == 2:
        #Centered difference (Parabolic)
        # it calculates velocity for n-1 frames!!!
        if time == 1:
            pass
        else:
            velocity= 0.5*(v_coo[2,:]-v_coo[0,:])
            speed = np.sqrt(velocity.dot(velocity))
            try:
                tg = velocity[1]/velocity[0]
                angle = np.arctan(tg)

            except:
                angle = pi/2

            vec_vel = np.vstack([vec_vel, velocity])
            vec_speed = np.vstack([vec_speed, speed])
            vec_angle = np.vstack([vec_angle, angle])

    else:
        print("Incorrect v degree, deg = 1, 2")
        
    # thetta = angle 
    # module_v = speed
    # vel = v

    return angle, speed, velocity