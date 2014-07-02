# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 11:25:45 2013

@author: Ben

New functions module to concatenate all the functions needed
"""
resistivity = 0. #default to zero resistivity
force_gpu = False #Default to using cpu for ffts

def eta_switch(switch=False):
    """
	Set the default value for eta
    """
    if(switch):
	global resistivity
	resistivity = 1.4E-7
    else:
	global resistivity
	resistivity = 0.

    return

def gpu_fft_on():
    """
    turn on gpu fft's
    """
    global force_gpu
    force_gpu = True

def gpu_fft_off():
    """
    turn off gpu ffts
    """
    global force_gpu
    force_gpu = False

def eta():
    """
    defines a constant, the resistivity
    """
    return resistivity

def read_data(file_in,dens_inc=False):
    """
    Reads in data from file_in.vtk into x, y, z, b, u
    """
    import vtk as vtk
    from vtk.util.numpy_support import vtk_to_numpy
    from numpy import rollaxis, reshape, array

    #Some vtk bookkeeping and object stuff
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(file_in)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    data = reader.GetOutput()
    dim = data.GetDimensions()
    nx = dim[0] - 1
    ny = dim[1] - 1
    nz = dim[2] - 1
    D = data.GetSpacing()
    N = (nx, ny, nz)

    #Read the raw data from the vtk file
    #u = vtk_to_numpy(data.GetCellData().GetArray('velocity')).reshape(nx, ny, nz, 3,order='F')
    #b = vtk_to_numpy(data.GetCellData().GetArray('cell_centered_B')).reshape(nx, ny, nz, 3,order='F') 

    ## #Convert to C-order and make a the 1st array index the vector component
    #b = array([b[...,0].ravel(order='F').reshape(nx,ny,nz), b[...,1].ravel(order='F').reshape(nx,ny,nz), b[...,2].ravel(order='F').reshape(nx,ny,nz)])
    #u = array([u[...,0].ravel(order='F').reshape(nx,ny,nz), u[...,1].ravel(order='F').reshape(nx,ny,nz), u[...,2].ravel(order='F').reshape(nx,ny,nz)])

    ## #Read the raw data from the vtk file
    u = vtk_to_numpy(data.GetCellData().GetArray('velocity')).reshape(nz,ny,nx,3)
    b = vtk_to_numpy(data.GetCellData().GetArray('cell_centered_B')).reshape(nz, ny, nx,3) 

    if(dens_inc):
        rho = vtk_to_numpy(data.GetCellData().GetArray('density')).reshape(nz,ny,nx)
        return (N,D,b.T,u.T,rho.T)
    #u = ascontiguousarray(array([u[...,0], u[...,1], u[...,2]]))
    #b = ascontiguousarray(array([b[...,0], b[...,1], b[...,2]]))
    #u = u.transpose(3,2,1,0)
    #b = b.transpose(3,2,1,0)
                        
    return (N, D, b.T, u.T)

def define_k(N, D):
    """
    Creates the 3d k arrays from N and D
    """
    from numpy.fft import fftfreq
    from numpy import array, pi, float32
    nx, ny, nz = N
    dx, dy, dz = D
    #Use a matlab like ndgrid to create the wave-vectors with the proper shape to take advantage of pythons vectorization
    kx, ky, kz = ndgrid(fftfreq(nx, dx)*2.*pi, fftfreq(ny, dy)*2.*pi, fftfreq(nz, dz)*2.*pi, dtype=float32)

    #Output a single vector
    return array([kx, ky, kz], dtype = float32)

def EMF(file_in, U=None, B=None, delt=False):
    """
    This function calculates the EMF (u x b). If u and b are already read in, pass them to the function to save time.
    """
    if U is None or B is None: #Check if U or B are defined, otherwise read them in
        N, D, U, B = read_data(file_in)

    if(delt): #Make the fluctuating components if needed
        u = delta(U)
        b = delta(B)
        return crossp(U, B), delta(crossp(U, B))

        #Old parts of the code, will need revising if utilized ****
        #return cross_p(U, B), delta(cross_p(u, b))
        #Test Moffat 1978:
        # e = -(u x <B>) - (<U> x b) - (u x b) + <u x b>
        #return cross_p(U, B), -cross_p(u, (B-b)) - cross_p((U-u), b) - cross_p(u, b) + delta(cross_p(u, b))
        #******
    else:
        return crossp(U, B)

def CurrentDensity(file_in, B=None, k=None, delt=False):
    """
    Function to calculate the electric current, curl(b) in Fourier Space. If b or k are defined can pass them to speed calculation.
    """
    if B is None or k is None: #Check if B or k are defined, else read in
        N,D,B,U = read_data(file_in)
        k = define_k(N,D)

    #Calculate J in k-space,\tilde{J} = ik \cdot \tilde{B}
    J = ifftvec(1.j*crossp(k, fftvec(B)))

    if(delt):
        return J, delta(J)    
    else:
        return J

def CurrentHelicity(file_in, J=None, B=None, k=None, delt=False):
    """
    Function to calculate the current helicity, j dot b. Faster if J and b are known.
    """
    # Check how much work we have to do
    if J is None:
        if B is None or k is None:
            N, D, B, U = read_data(file_in)
            k = define_k(N, D)                   
        J = CurrentDensity(file_in, B, k)    
    if(delt):
        b = delta(B)
        jc = delta(J)
        return dot_p(J, B), LargeScale(dot_p(jc, b))
    else:
        return dot_p(J, B)

def helicity(u,k):
    """
    Function to calculate the fluid helicity
    w = u \cdot \nabla \times u
    """
    return dot_p(u, ifftvec(1.j*crossp(k,fftvec(u))))

def vorticity(u,k):
	"""
	Function which calculates the vorticity in Fourier Space
	"""
	return ifftvec(1.j*crossp(k,fftvec(u)))

def Potential(file_in, J=None, E=None, k=None, D=None, N=None, delt=False):
    """
    Function to calculate the vector and scalar potentials using the spectral_poisson3d function
    Spectral_poisson3d takes a source term (scalar, 3d) and the grid spacing as a tuple and returns the potential
    """
    from numpy import array

    #Check for defined variables, read in or calculate as needed
    if D is None or J is None or E is None:
        N,D,B,U = read_data(file_in)

    if k is None:
        k = define_k(N,D)
    
    #Call spectral_poisson3d to set the potentials and return
    if E is None:
        E = EMF(file_in, U, B)
        
    if J is None:
        J = CurrentDensity(file_in,B,k)

    #Make a 4-vector [phi, Ax, Ay, Az] by calling the Poisson solver for each component
    # The solver takes care of the fft/ifft if fft_source is True
    A = array([spectral_poisson3d(-1.j*dot_p(k,fftvec(E)),D), #Phi = (ik \cdot E)/k^2   --There is a - sign in the poisson solver
               spectral_poisson3d(J[0],D,fft_source=True),    #Ax  = -Jx/k^2
               spectral_poisson3d(J[1],D,fft_source=True),    #Ay  = -Jy/k^2
                spectral_poisson3d(J[2],D,fft_source=True)    #Az  = -Jz/k^2
                ])

    if delt:
        a = array([delta(A[0]), delta(A[1]), delta(A[2]), delta(A[3])])
        return A, a
    else:        
        return A

def MagneticHelicity(file_in, A=None, B=None, delt=False, large=False):
    """
    Function to calculate the magnetic helicity, a dot b. Faster if a and b are known.
    """
    from numpy import copy
    #Again, check and see how much work is already done
    if B is None:
        N, D, B, U = read_data(file_in)
    
    if A is None:
        A = Potential(file_in)

    A = A[1:].copy()


    #H is a simple dot product
    #H = A \cdot B, <H> = <A> \cdot <B>, h = <a \cdot b>  
    if(delt):
        b = delta(B)
        a = delta(A)
        if(large):
            #return dot_p(A, B), dot_p((A-a), (B-b)), dot_p(A, B) - dot_p((A-a), (B-b))
            return dot_p(A, B), dot_p((A-a), (B-b)), LargeScale(dot_p(a, b))
        else:
            return dot_p(A, B), LargeScale(dot_p(a, b))
    else:
        return dot_p(A, B)

def MagneticHelicityFlux(file_in, A=None, E=None, delt=False, large=False):
    """
    Function to calculate the flux of magnetic helicity
    """
    from numpy import copy
    if(delt):
        N, D, B, U = read_data(file_in)
        k = define_k(N,D)
        E, e = EMF(file_in, U, B, delt=True)
        b = delta(B)
        Jc, jc = CurrentDensity(file_in, B, k, delt = True)

            
        if A is None:
            A, a = Potential(file_in, delt=True)
        else:
            a = A.copy()
            a[0] = delta(A[0])
            a[1:] = delta(A[1:])
            
        #Split the 4-vector into the scalar and vector potentials
        Phi = A[0].copy()
        A = A[1:].copy()
        phi = a[0].copy()
        a = a[1:].copy()


        #Total:
        #J_H = A \times E + \Phi B + \eta J \times A
		#Small scale:
        #J_h = < a \times e > + <phi b> + <\eta j \times a>
		#Large scale:
        #<J_H> = <A> \times <E> + <\Phi B> + \eta <J> \times <A> 

        Jh = crossp(A, E) + Phi*B + eta()*crossp(Jc, A)
        Jh_small = LargeScale(crossp(a, e)) + LargeScale(phi*b) + LargeScale(eta()*crossp(jc, a))
        #Jh_small = crossp(a, e) + phi*b + eta()*crossp(jc, a)
        if(large):
			#Returns, Total, Small Scale, Large Scale
            return Jh, Jh_small, crossp((A-a), (E-e)) + LargeScale(Phi*B) + eta()*crossp((Jc-jc), (A-a))
        else:
            return Jh, Jh_small
    else:
        if E is None:
            N, D, B, U = read_data(file_in)
            E = EMF(file_in, U, B)
        if A is None:
            A, Phi = Potential(file_in)
            Phi = A[0].copy()
            A = A[1:].copy()
        return Jh 

##Functions to manipulate data

def spectral_poisson3d(source,h,fft_source=False):
    """
    Solves the 3D Poisson equation using ffts
    Input the source term and a tuple of the grid spacing
    EX:
        phi = spectral_poisson3d(rho, (dx,dy,dz))
    """
    from numpy.fft import fftfreq
    from numpy import pi
    nx, ny, nz = source.shape
    hx, hy, hz = h

    #Define the k-vector, in retrospect, I could have used the function for this
    kx, ky, kz = ndgrid(fftfreq(nx,hx)*2.*pi, fftfreq(ny,hy)*2.*pi, fftfreq(nz,hz)*2.*pi)
    k2 = -(kx**2 + ky**2 + kz**2)
    k2[0,0,0] = 1. #Set the DC component gain = 1 (will subtract this off later anyway)
    if fft_source: #If the source needs to be fft'ed
        V = fftvec(-source)/k2
    else: #Otherwise, just divide by k^2
        V = -source/k2
    V[0,0,0] = 0. #Set the DC component to 0, effective subtracts the mean from the solution
                  #Forces a unique solution to the periodic BC Poisson problem
    return ifftvec(V)

def fftvec(vec):
    """
    performs a fft on a vector with 3 components in the first index position
    This is really just a wrapper for fft, fftn and their inverses
    """
    try:
        from anfft import fft, fftn
        fft_type = 1
    except:
#        print "Could not import anfft, importing scipy instead."
#Update 9/18/2013 -- numpy with mkl is way faster than scipy
        import mkl
        mkl.set_num_threads(8)
        from numpy.fft import fft, fftn
        fft_type = 0
        
    if force_gpu:
        fft_type = 2 #set gpu fft's manually -- not sure what a automatic way would look like
    
    from numpy import complex64, shape, array, empty
    if vec.ndim > 2:
        if vec.shape[0] == 3:
            # "Vector": first index has size 3 so fft the other columns
            if fft_type==1:
                return array([fftn(i,measure=True) for i in vec]).astype(complex64)
#                result = empty(vec.shape, dtype=complex64)
#                result[0] = fftn(vec[0], measure=True).astype(complex64)
#                result[1] = fftn(vec[1], measure=True).astype(complex64)
#                result[2] = fftn(vec[2], measure=True).astype(complex64)
#                return result
                
            elif fft_type==0:
                return fftn(vec, axes=range(1,vec.ndim)).astype(complex64)
            elif fft_type==2:
#                return array([gpu_fft(i) for i in vec.astype(complex64)])
                result = empty(vec.shape, dtype=complex64)
                result[0] = gpu_fft(vec[0].copy())
                result[1] = gpu_fft(vec[1].copy())
                result[2] = gpu_fft(vec[2].copy())
                return result
        else: # "Scalar", fft the whole thing
            if fft_type==1:
                return fftn(vec,measure=True).astype(complex64)
            elif fft_type==0:
                return fftn(vec).astype(complex64)
            elif fft_type==2:
                return gpu_fft(vec.copy())
    elif vec.ndim == 1: #Not a vector, so use fft
        if fft_type==1:
            return fft(vec,measure = True).astype(complex64)
        elif fft_type==0:
            return fft(vec).astype(complex64)
        elif fft_type==2:
            return gpu_fft(vec.astype(complex64))
    else:
        #0th index is 3, so its a vector
        #return fft(vec, axis=1).astype(complex64)
        return array([fft(i) for i in vec])
        
def ifftvec(vec):
    """
    performs a fft on a vector with 3 components in the last index position
    This is a wrapper for ifft and ifftn
    """
    try:
        from anfft import ifft, ifftn
        fft_type = 1
    except:
#        print "Could not import anfft, importing scipy instead."
#Update 9/18/2013 -- numpy with mkl is way faster than scipy
        import mkl
        mkl.set_num_threads(8)
        from numpy.fft import ifft, ifftn
        fft_type = 0
        
    if force_gpu:
        fft_type = 2 #set gpu fft's manually -- not sure what a automatic way would look like
                  
    from numpy import float32, real, array, empty, complex64
    if vec.ndim > 2:
        if vec.shape[0] == 3:
            # "Vector": first index has size 3 so fft the other columns
            if fft_type==1:
                return array([ifftn(i,measure=True) for i in vec]).astype(float32)
#                result = empty(vec.shape, dtype=float32)
#                result[0] = real(ifftn(vec[0], measure=True)).astype(float32)
#                result[1] = real(ifftn(vec[1], measure=True)).astype(float32)
#                result[2] = real(ifftn(vec[2], measure=True)).astype(float32)
#                return result
            elif fft_type==0:
                return ifftn(vec, axes=range(1,vec.ndim)).astype(float32)
            elif fft_type==2:
#                return array([gpu_ifft(i) for i in vec]).astype(float32)
                result = empty(vec.shape, dtype=float32)
                result[0] = gpu_ifft(vec[0].copy()).astype(float32)
                result[1] = gpu_ifft(vec[1].copy()).astype(float32)
                result[2] = gpu_ifft(vec[2].copy()).astype(float32)
                return result
        else: # "Scalar", fft the whole thing
            if fft_type==1:
                return ifftn(vec,measure=True).astype(float32)
            elif fft_type==0:
                return ifftn(vec).astype(float32)
            elif fft_type==2:
                return gpu_ifft(vec.copy()).astype(float32)
    elif vec.ndim == 1: #Not a vector, so use fft
        if fft_type==1:
            return ifft(vec,measure = True).astype(float32)
        elif fft_type==0:
            return ifft(vec).astype(float32)
        elif fft_type==2:
            return gpu_ifft(vec).astype(float32)
    else:
        #0th index is 3, so its a vector
        #return fft(vec, axis=1).astype(complex64)
        return array([ifft(i) for i in vec]).astype(float32)


def gpu_fft(vec):
    """
    Uses the pyopencl and pyfft libraries to perform an fft on the GPU
    """
    from pyfft.cl import Plan as cl_plan
    import pyopencl as cl
    import pyopencl.array as cl_array
    from numpy import complex64, shape, float32, complex128
    print vec.shape
    array_size = vec.shape
    #Find the GPU's available
    platform = cl.get_platforms()
    my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
    #Create a context using the GPU's found in the above step    
    ctx = cl.Context(devices=my_gpu_devices)
    #Create queue using that context
    queue = cl.CommandQueue(ctx)
    #create a plan based on the size of the array
    #Make a temporary copy of vec so that things don't get all messed up
    plan = cl_plan(array_size, dtype=complex64, queue=queue)
    #    plan = cl_plan(array_size,queue=queue)
      
    alloc = cl.tools.ImmediateAllocator(queue)
    cl.tools.MemoryPool(alloc).stop_holding()
 
    #temp = vec.copy().astype(complex64)
    #gpu_data = cl_array.to_device(queue, temp)
    vec = vec.astype(complex64)
    
    cl.enqueue_barrier(queue)
    gpu_data = cl_array.to_device(queue, vec, allocator = alloc, async = True)
    gpu_data.queue.finish()    
    cl.enqueue_barrier(queue)    
    plan.execute(gpu_data.data)    
    cl.enqueue_barrier(queue)

    ans = gpu_data.get()
    gpu_data.data.release()
    gpu_data.queue.finish()
    queue.flush()
    for i in range(20):
        pass
    return ans

def gpu_ifft(vec):
    """
    Uses the pyopencl and pyfft libraries to perform an fft on the GPU
    """
    from pyfft.cl import Plan as cl_plan
    import pyopencl as cl
    import pyopencl.array as cl_array
    from numpy import complex64, shape, complex128, float32, real    

    array_size = vec.shape
    #Find the GPU's available
    platform = cl.get_platforms()
    my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
    #Create a context using the GPU's found in the above step    
    ctx = cl.Context(devices=my_gpu_devices)
    #Create queue using that context
    queue = cl.CommandQueue(ctx)    
    #    plan = cl_plan(array_size,queue=queue)
    #Make a temporary copy of vec so that things don't get all messed up
    
    ##temp = vec.copy().astype(complex64)
    
    plan = cl_plan(array_size, dtype=complex64, queue=queue)
    #    plan = cl_plan(array_size,queue=queue)
            
    alloc = cl.tools.ImmediateAllocator(queue)
    cl.tools.MemoryPool(alloc).stop_holding()
    ##gpu_data = cl_array.to_device(queue, temp)
    vec = vec.astype(complex64)
         
    cl.enqueue_barrier(queue)
    gpu_data = cl_array.to_device(queue, vec, allocator = alloc, async = True)
    gpu_data.queue.finish()
    
    cl.enqueue_barrier(queue)
    plan.execute(gpu_data.data, inverse=True)
    cl.enqueue_barrier(queue)

    ans = gpu_data.get()
    gpu_data.data.release()
    gpu_data.queue.finish()
    queue.flush()
    for i in range(20):
        pass    
    return ans

def power_spectrum(x):
    """
    Calculates the power spectrum of a (nx,ny,nz) array as a function of kz
    """
    from numpy import conjugate, shape
    nx,ny,nz = x.shape
    x_ = fftvec(x)
    x_[0,0,0] = 0.
    return sum(sum(x_*conjugate(x_),0),0)[:nz/2]/(nx*ny*nz)

def weighted_power_spectrum(x,kz):
    """
    Calculates the power spectrum times kz
    """
    from numpy import conjugate, shape
    nx,ny,nz = x.shape
    x_ = fftvec(x)
    x_[0,0,0] = 0.
    return sum(sum(x_*conjugate(x_),0),0)[:nz/2]/(nx*ny*nz)*kz[:nz/2]
        
def zaverage(a):
    """
    Takes a 3D array and computes the average over the x-y plane
    If the first index has size 3, its a vector so output will be a vector
    """
    from numpy import mean
    if a.shape[0] == 3:
        return mean(mean(a, -2), -2)
    else:   
        return mean(mean(a, 0), 0)

def lowpass(a, window='hamming', alpha=0.1, fc = 1./64.):
    """
    Function which takes the num_ks smallest k-value positions in each dimension and filters the rest.
    The input is expected to be of the form a[3,...] for a vector.
    Alpha defines the sharpness of the filter window if hamming is chosen
    fc is the cutoff frequency relative to the nyquist frequency
    """
    from numpy import shape,ones,zeros,float32
    #Does the filtering in 1 line
    #    -FFT
    #    -Window the data
    #    -IFFT
    
    #Create a rectangular window
    if(window=='rectangular'):
        return ifftvec(create_filter(a.shape, fc,alpha, rect=True)*fftvec(a))
    
    if(window=='hamming'):#Or create a "hamming" window
        return ifftvec(create_filter(a.shape, fc, alpha)*fftvec(a))

def highpass(a, window='hamming', alpha=0.1, fc=1./64.):
    """
    This function does the same sort of thing as lowpass, except it passes the high k components set by fc
    """
    from numpy import shape,ones,float32
    #Does the filtering in 1 line
    #    -FFT
    #    -Window the data
    #    -IFFT

    #create_filter creates a lowpass window, so need to do a 1 - filter
    #Create a rectangular window
    if(window=='rectangular'):
        return ifftvec((1.-create_filter(a.shape, fc, alpha,rect=True))*fftvec(a))
    
    if(window=='hamming'):#Or create a "hamming" window
        return ifftvec((1.-create_filter(a.shape, fc, alpha))*fftvec(a))

def tukey_filter(width, alpha):
    """
    Creates a window with length width and sharpness alpha
    1 is a cosine (Hamming)
    0 is a rectangle/boxcar/brickwall (passing 0 will actually give a divide by zero, so don't do that)
    """
    from numpy import arange, ones, cos, pi, float32
    #create the x values to pass to the function
    x = arange(width).astype(float32)
    #Do some fancy slicing with numpy arrays to create a piecwise function
    p1 = slice(None, int(alpha*(width-1)/2))
    p2 = slice(int(alpha*(width-1)/2), int((width-1)*(1-alpha/2)))
    p3 = slice( int((width-1)*(1-alpha/2)), int(width-1) )
    #Set default values to 1
    result = ones(width)
    #Create the piecewise function
    result[p1] = 0.5*(1.+cos(pi*(2.*x[p1]/alpha/(width-1)-1.)))
    result[p2] = 1.
    result[p3] = 0.5*(1.+cos(pi*(2.*x[p3]/alpha/(width-1)-alpha/2.+1.)))
    if width%2 == 0:
        result[:-width/2:-1] = result[:width/2-1] #mirror the window to work in k-space
    else:
        result[:-width/2:-1] = result[:width/2]
    return result

def create_filter(axis_dimensions, fc, alpha, rect=False):
    """
    Creates the 3D filter mask to be multiplied by the signal
    """
    from numpy import append, insert, ones, zeros, hstack, float32

    #If its a vector, only look at 1: indeces
    if axis_dimensions[0] == 3:
        ad = axis_dimensions[1:]
    else:
        ad = axis_dimensions

    fcx = int(ad[0]/2*fc)
    fcy = int(ad[1]/2*fc)
    fcz = int(ad[2]/2*fc)

    if(rect):
#        filter = zeros(ad,dtype=float32)
#        filter[:1,:,:]=1.
#        filter[:,:4,:]=1.
#        filter[:,:,:4]=1.
#        filter[-1:,:,:]=1.
#        filter[:,-4:,:]=1.
#        filter[:,:,-4:]=1.
#        return filter

        f1 = zeros(ad[0])
        f2 = zeros(ad[1])
        f3 = zeros(ad[2])
        f1[:2]=1.
        f2[:8]=1.
        f3[:8]=1.
        f1[-2:]=1.
        f2[-8:]=1.
        f2[-8:]=1.
#        f1 = zeros(ad[0]-fcx)
#        f2 = zeros(ad[1]-fcy)
#        f3 = zeros(ad[2]-fcz)
#        f1 = hstack((ones(fcx/2),f1,ones(fcx/2)))
#        f2 = hstack((ones(fcy/2),f2,ones(fcy/2)))
#        f3 = hstack((ones(fcz/2),f3,ones(fcz/2)))
    else:
        #Create the 1D filters in each dimension
        f1 = 1.-tukey_filter(ad[0]-fcx, alpha)
        f2 = 1.-tukey_filter(ad[1]-fcy, alpha)
        f3 = 1.-tukey_filter(ad[2]-fcz, alpha)
        #Pad the ones on the ends up to the cutoff desired
        f1 = hstack((ones((ad[0]-f1.size)/2),f1,ones((ad[0]-f1.size)/2)))
        f2 = hstack((ones((ad[1]-f2.size)/2),f2,ones((ad[1]-f2.size)/2)))
        f3 = hstack((ones((ad[2]-f3.size)/2),f3,ones((ad[2]-f3.size)/2)))

    if f1.size != ad[0]:
        f1 = insert(f1, ad[0]/2, 0.)

    if f2.size != ad[1]:
        f2 = insert(f2, ad[1]/2, 0.)

    if f3.size != ad[2]:
        f3 = insert(f3, ad[2]/2, 0.)
    
    #This effectively does an outer product to create the array mask
    f1, f2, f3 = ndgrid(f1,f2,f3)

    return f1*f2*f3 #This is the actual outer product to make the mask

def dot_p(a, b):
    """
    Custom dot product function for (3, nx, ny, nz) sized arrays
    DOES NOT COMPLEX CONJUGATE
    *** Beware *** No checking is done about the size of the input arrays, you've been warned
    This should actually work for any array with 3 as the first index deonoting vector components.
    So long as they are the same shape (or can broadcast)
    """
    #Because of the way python-numpy arrays work, if the input are arrays, this is the sames as:
    # b[0,...]*a[0,...] etc.
    return sum(i*j for i,j in zip(a,b))
    #return b[0]*a[0] + b[1]*a[1] + b[2]*a[2]

def crossp(a, b):
    """
    Custom cross product routine for (3, nx, ny, nz) sized arrays
    The same warnings as the dot_p function apply. Interestingly, I could also make this a wrapper for
    numpy.cross(a, b, axis=0) as it does the same thing
    """
    from numpy import array
    return array([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]])

def mag(x):
    """
    Computes the vector magnitude
    """
    from numpy import sqrt
    return sqrt(sum(i**2 for i in x))

def LargeScale(x):
    """
    Takes a lowpass of the data
    """
    #return lowpass(x,window='rectangular',fc=8./64.)
    #return lowpass(x,window = 'rectangular', fc=1./64.)
    #return lowpass(x,window='hamming',alpha=0.001,fc=1./64.)
    return lowpass(x,window='rectangular',alpha=0.1,fc=1./64.)

def delta(x):
    """
    Takes effectively a highpass of the data
    a = A - <A>
    """
    #return x - LargeScale(x)
    #return highpass(x,window='hamming',alpha=0.001,fc=1./64.)
    return highpass(x,window='rectangular',alpha=0.1,fc=1./64.)

def dh_dt(files_in,dt,back=False):
    """
    Takes a 5 point stencil time derivative of h (Note, small scale only)
    """
    from numpy import array
    if len(files_in) == 4:
        #co = array([1./12./dt, -2./3./dt, +2./3./dt, -1./12./dt])
        co = array([-3./10./dt, -1./10./dt, +1./10./dt, 3./10./dt])
    else:
        if back:
            co = array([-1., 1.])/dt
        else:
            co = array([-1./2./dt, 1./2./dt])
        
    dh = 0.
    dH = 0.
    for x,i in zip(files_in,range(len(files_in))):
        H, h= MagneticHelicity(x,delt=True)
        dh += co[i]*h
        dH += co[i]*H
    return dh, dH

def ndgrid(*args, **kwargs): 
    """ 
    n-dimensional gridding like Matlab's NDGRID 
    
    The input *args are an arbitrary number of numerical sequences, 
    e.g. lists, arrays, or tuples. 
    The i-th dimension of the i-th output argument 
    has copies of the i-th input argument. 
    
    Optional keyword argument: 
    same_dtype : If False (default), the result is an ndarray. 
                 If True, the result is a lists of ndarrays, possibly with 
                 different dtype. This can save space if some *args 
                 have a smaller dtype than others. 

    Typical usage: 
    >>> x, y, z = [0, 1], [2, 3, 4], [5, 6, 7, 8] 
    >>> X, Y, Z = ndgrid(x, y, z) # unpacking the returned ndarray into X, Y, Z 

    Each of X, Y, Z has shape [len(v) for v in x, y, z]. 
    >>> X.shape == Y.shape == Z.shape == (2, 3, 4) 
    True 
    >>> X 
    array([[[0, 0, 0, 0], 
            [0, 0, 0, 0], 
            [0, 0, 0, 0]], 
           [[1, 1, 1, 1], 
            [1, 1, 1, 1], 
            [1, 1, 1, 1]]]) 
    >>> Y 
    array([[[2, 2, 2, 2], 
            [3, 3, 3, 3], 
            [4, 4, 4, 4]], 
           [[2, 2, 2, 2], 
            [3, 3, 3, 3], 
            [4, 4, 4, 4]]]) 
    >>> Z 
    array([[[5, 6, 7, 8], 
            [5, 6, 7, 8], 
            [5, 6, 7, 8]], 
           [[5, 6, 7, 8], 
            [5, 6, 7, 8], 
            [5, 6, 7, 8]]]) 
    
    With an unpacked argument list: 
    >>> V = [[0, 1], [2, 3, 4]] 
    >>> ndgrid(*V) # an array of two arrays with shape (2, 3) 
    array([[[0, 0, 0], 
            [1, 1, 1]], 
           [[2, 3, 4], 
            [2, 3, 4]]]) 
    
    For input vectors of different data types, same_dtype=False makes ndgrid() 
    return a list of arrays with the respective dtype. 
    >>> ndgrid([0, 1], [1.0, 1.1, 1.2], same_dtype=False) 
    [array([[0, 0, 0], [1, 1, 1]]), 
     array([[ 1. ,  1.1,  1.2], [ 1. ,  1.1,  1.2]])] 
    
    Default is to return a single array. 
    >>> ndgrid([0, 1], [1.0, 1.1, 1.2]) 
    array([[[ 0. ,  0. ,  0. ], [ 1. ,  1. ,  1. ]], 
           [[ 1. ,  1.1,  1.2], [ 1. ,  1.1,  1.2]]]) 
    """ 
    from numpy import array, zeros, ones_like, append, shape
    same_dtype = kwargs.get("same_dtype", True) 
    V = [array(v) for v in args] # ensure all input vectors are arrays 
    shape = [len(v) for v in args] # common shape of the outputs 
    result = [] 
    for i, v in enumerate(V): 
        # reshape v so it can broadcast to the common shape 
        # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        zero = zeros(shape, dtype=v.dtype) 
        thisshape = ones_like(shape) 
        thisshape[i] = shape[i] 
        result.append(zero + v.reshape(thisshape)) 
    if same_dtype: 
        return array(result) # converts to a common dtype 
    else: 
        return result # keeps separate dtype for each output 



### These functions are sort of kludges, they take time averages etc and save binary files to disk to conserve memory

# Function to take time averages and save data to disk
def gen_t_averages():
    """
    Take in all the time slices and average them, this function needs to be edited to produce the right files
    I think a glob.glob might be appropriate here...
    """
    from numpy import array
    #fp is the path to the directory where the time slices are kept
    #fp = '/1/home/jackelb/Research/AthenaDumps/AthenaDumps/strat64z6/'
    #fp = '/1/home/jackelb/Research/AthenaDumps/AthenaDumps/128/'
    fp = '/1/home/jackelb/Research/AthenaDumps/AthenaDumps/strat128z4/'
    
    #ts are the time stamps of the individual files in the syntax: _xxxx
    #ts = array(['_1160', '_1164', '_1168', '_1172', '_1176', '_1180'])
    #ts = array(['_0101','_0102','_0103','_0104'])
    ts = array(['_1229','_1230','_1231','_1232'])
    ##ts = array(['_1180', '_1181', '_1182'])

    #t_slice is the name of the time slice files with timestamp
    #t_slice = array(['Strat.1160.vtk', 'Strat.1164.vtk', 'Strat.1168.vtk', 'Strat.1172.vtk', 
    #                 'Strat.1176.vtk', 'Strat.1180.vtk'])
    #t_slice = array(['Strat.0101.vtk', 'Strat.0102.vtk', 'Strat.0103.vtk', 'Strat.0104.vtk'])
    t_slice = array(['Strat.1229.vtk', 'Strat.1230.vtk', 'Strat.1231.vtk', 'Strat.1232.vtk'])    
    ##t_slice = array(['Strat.1180.vtk', 'Strat.1181.vtk', 'Strat.1182.vtk'])

    #make the variable files
    remake = True
    if(remake==True):
        for i in range(len(ts)):
            make_variable_files(fp, t_slice[i], ts[i])
		
        #make the time averages
    qs = array(['B', 'U', 'E', 'J', 'H', 'Hc', 'H_t', 'JH', 'Potential'])
    #ts = array(['_1160.npy', '_1164.npy', '_1168.npy', '_1172.npy', '_1176.npy', '_1180.npy'])
    #ts = array(['_0101.npy','_0102.npy','_0103.npy','_0104.npy'])
    ts = array(['_1229.npy','_1230.npy','_1231.npy','_1232.npy'])
   
    ##ts = array(['_1180.npy', '_1181.npy', '_1182.npy'])
	
    for i in range(len(qs)):
        time_average(fp, qs[i], ts)

def make_variable_files(file_prefix, time_stamp, file_suffix):
    """
    Outputs numpy files for all the variables
    """
    from numpy import save
    file_in = file_prefix+time_stamp
    
    print 'Reading from', file_in
    N, D, B, U = read_data(file_in)
    k = define_k(N, D)
    b = B - delta(B)
    u = U - delta(U)
    
    print 'Writing', file_prefix + 'B' + file_suffix
    save(file_prefix+'B'+file_suffix, (B, b))
    print 'Writing', file_prefix + 'U' + file_suffix
    save(file_prefix+'U'+file_suffix, (U, u))
    #u = []
    #b = []

    E, e = EMF(file_in, U, B, delt=True)
    print 'Writing', file_prefix + 'E' + file_suffix
    save(file_prefix+'E'+file_suffix, (E, e))
    
    J, jc = CurrentDensity(file_in, B, k, delt=True)
    print 'Writing', file_prefix + 'Jc' + file_suffix
    save(file_prefix+'J'+file_suffix, (J, jc))
    H_t = dot_p( 2.*LargeScale(B), LargeScale(crossp(u, b)))
    print 'Writing', file_prefix + 'H_t' + file_suffix
    save(file_prefix+'H_t'+file_suffix, (H_t))
    jc = []
    b = []
    H_t = []
    e = []
    u = []
    
    Hc, hc = CurrentHelicity(file_in, J, B, k, delt=True)
    print 'Writing', file_prefix + 'Hc' + file_suffix
    save(file_prefix+'Hc'+file_suffix, (Hc, hc))
    Hc = []
    hc = []

    Pot_4, pot_4 = Potential(file_in, J, E, k, delt=True)
    print 'Writing', file_prefix + 'Potential' + file_suffix
    save(file_prefix+'Potential'+file_suffix, (Pot_4, pot_4))
    pot_4 = []
    J = []
    
    J_H, J_h, J_h_alt = MagneticHelicityFlux(file_in, Pot_4, E, delt=True, large=True)
    print 'Writing', file_prefix + 'Jh' + file_suffix
    save(file_prefix+'JH'+file_suffix, (J_H, J_h, J_h_alt))

    J_H = []
    J_h = []
    J_h_alt
    E = []

    H, h = MagneticHelicity(file_in, Pot_4, B, delt=True)
    print 'Writing', file_prefix + 'H' + file_suffix
    save(file_prefix+'H'+file_suffix, (H, h))
    return

def time_average(file_prefix, quantity, time_stamp):
    """
    Function to take the time average of a set of data
    """
    from numpy import zeros, float32, load, mean, save
    if(quantity=='JH'):
        print 'Loading data from', file_prefix+quantity
        JH,Jh,Jh_alt = load(file_prefix+quantity+time_stamp[0])
        JH = zeros(JH.shape + time_stamp.shape, dtype=float32)
        Jh = zeros(JH.shape, dtype=float32)
        Jh_alt = zeros(JH.shape, dtype=float32)
        for i in range(len(time_stamp)):
            JH[...,i], Jh[...,i], Jh_alt[...,i] = load(file_prefix+quantity+time_stamp[i])
        save(file_prefix+quantity, (mean(JH, -1), mean(Jh, -1), mean(Jh_alt, -1)))
    elif(quantity=='H_t'):
        print 'Loading data from', file_prefix+quantity
        H_t = load(file_prefix+quantity+time_stamp[0])
        H_t = zeros(H_t.shape + time_stamp.shape, dtype=float32)
        for i in range(len(time_stamp)):
            H_t[..., i] = load(file_prefix+quantity+time_stamp[i])
        save(file_prefix+quantity, mean(H_t, -1))

    else:  
        #print 'Loading data from', file_prefix+quantity
        Q, q = load(file_prefix+quantity+time_stamp[0])
        Q = zeros(Q.shape+time_stamp.shape, dtype=float32)
        q = zeros(q.shape+time_stamp.shape, dtype=float32)
        for i in range(len(time_stamp)):
            print 'Loading', file_prefix+quantity+time_stamp[i]
            Q[..., i], q[..., i] = load(file_prefix+quantity+time_stamp[i])
        save(file_prefix+quantity, (mean(Q, -1), mean(q, -1)))
        
        
##Plotting Functions
def plot_vec(x, y, titlear, x_label, y_label, filename):
    """
    Function which plots a vector in 3 side by side plots
    """
    from matplotlib.pyplot import figure, gcf, subplot, plot, title, xlabel, ylabel, tight_layout, savefig, close
    figure
    gcf().set_size_inches((8, 18.))
    for ii in range(3):
        subplot(3, 1, ii+1)
        plot(x, y[ii], '+')
        title(titlear[ii])
        xlabel(x_label[ii])
        ylabel(y_label[ii])
    tight_layout()
    savefig(filename)
    close()
    return

def plot_2vec(x, y1, y2, titles, xlabels, ylabels, filename):
    """
    Function which plots the averages and lowpass filters
    """
    from matplotlib.pyplot import figure, gcf, subplot, plot, title, xlabel, ylabel, tight_layout, savefig, close, gca
    figure
    gcf().set_size_inches((8.5, 11))
    for ii in range(3):
        subplot(3, 2, 2*ii+1)
        plot(x, y1[ii], '+')
        title(titles[ii])
        xlabel(xlabels[ii])
        ylabel(ylabels[ii])
        xa, xb = gca().get_xlim()
        ya, yb = gca().get_ylim()
        gca().set_aspect((xb-xa)/(yb-ya))
        subplot(3, 2, (ii+1)*2)
        plot(x, y2[ii], '+')
        title(titles[ii+3])
        xlabel(xlabels[ii+3])
        ylabel(ylabels[ii+3])
        xa, xb = gca().get_xlim()
        ya, yb = gca().get_ylim()
        gca().set_aspect((xb-xa)/(yb-ya))
        
    tight_layout()
    savefig(filename)
    close()
    return

