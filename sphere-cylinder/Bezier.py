import numpy as np
import constants as c
import matplotlib.pyplot as plt

# defiening Bezier curves of degree n based on the recursive definition
class Bezier():
    
    # returns a point between P1 and P2, parametised by t.
    def TwoPoints(t, P1, P2):
        out = (1 - t) * P1 + t * P2
        return out
    
    # returns a list of points as numpy arrays, interpolated by the Bezier curve definition
    def Points(t, points):
        newpoints = []
        for i1 in range(0, len(points) - 1):
            newpoints += [Bezier.TwoPoints(t, points[i1], points[i1 + 1])]
        return newpoints
    
    # returns a point interpolated by the Bezier equation
    def Point(t, points):
        newpoints = points
        while len(newpoints) > 1:
            newpoints = Bezier.Points(t, newpoints)
        return newpoints[0]
    
    # returns a numpy array of points, t_values a list of ints/floats
    def Curve(t_values, points):
        curve = np.array([[0.0] * len(points[0])])
        for t in t_values:
            curve = np.append(curve, [Bezier.Point(t, points)], axis=0)
        curve = np.delete(curve, 0, 0)
        return curve
    
    # simplest discretization: returns a grid of ones and zeros, ones are where the fiber is
    def DistCurve(points):
        x = points[:, 0].astype(int)
        y = points[:, 1].astype(int)
        grid = np.zeros((c.GRID_SIZE, c.GRID_SIZE))
        for i in range(len(points)):
            grid[y[i], x[i]] = 1
        return grid
            
    # rasterization by Bresenham algorithm
    def plotLine(x0, y0, x1, y1):
        grid = np.zeros((20, 20))
        x_array = np.empty((1, 100))
        y_array = np.empty((1, 100))
        dx =  abs(x1-x0)
        if x0<x1:
            sx = 1 
        else:
            sx = -1
        
        dy = -abs(y1-y0)
        if y0<y1:
            sy = 1 
        else:
            sy = -1
        #/* error value e_xy */
        err = dx+dy
        e2 = 0 
         
        while(True):
            np.append(x_array, x0)
            np.append(y_array, y0)
            #print(x0)
            grid[x0, y0] = 1
            if (x0==x1 & y0==y1):
                break
            e2 = 2*err
            if (e2 >= dy):
                err += dy
                x0 += sx #/* e_xy+e_x > 0 */
                
            if (e2 <= dx):
                err += dx
                y0 += sy #/* e_xy+e_y < 0 */
        return grid

    def plotQuadBezierSeg(x0, y0, x1, y1, x2, y2):
        sx = x2-x1
        sy = y2-y1
        
        #relative values for checks
        xx = x0-x1
        yy = y0-y1
        xy = 0
        #/* curvature */     
        dx=0
        dy=0
        err=0 
        cur = xx*sy-yy*sx                
        
        #/* sign of gradient must not change */
        if xx*sx <= 0 & yy*sy <= 0:
            print("OK")
        
        # { /* begin with longer part */ 
        if (sx*sx+sy*sy > xx*xx+yy*yy): 
            # /* swap P0 P2 */
            x2 = x0
            x0 = sx+x1
            y2 = y0
            y0 = sy+y1
            cur = -cur
        #  /* no straight line */
        if (cur != 0): 
            # #/* x step direction */                                   
            xx += sx
            if (x0 < x2):
                sx =  1
            else:
                sx = -1
            xx *= sx
            
            #  #/* y step direction */
            yy += sy
            if(y0 < y2):
                sy = 1 
            else:
                sy = -1
            yy *= sy
            
            # #/* differences 2nd degree */
            xy = 2*xx*yy
            xx *= xx
            yy *= yy
            
            #  # /* negated curvature? */
            if (cur*sx*sy < 0):                      
                xx = -xx
                yy = -yy
                xy = -xy
                cur = -cur
        
        # /* differences 1st degree */
        dx = 4.0*sy*cur*(x1-x0)+xx-xy             
        dy = 4.0*sx*cur*(y0-y1)+yy-xy
         #/* error 1st step */
        xx += xx
        yy += yy
        err = dx+dy+xy       
          
        # /* gradient negates -> algorithm fails */
        while (dy < dx ):
            
            # /* plot curve */
            #grid[x0, y0] = 1
            #np.append(x_array, x0)
            #np.append(y_array, y0)
            #print(x0)
            #print(y0)
            # /* last pixel -> curve finished */                                   
            if (x0 == x2 & y0 == y2):
                return
            # /* save value for test of y step */
            y1 = 2*err < dx   
            # #/* x step */              
            if (2*err > dy):
                x0 += sx
                dx -= xy
                dy += yy
                err += dy
            # /* y step */
            if y1:
                y0 += sy
                dy -= xy
                dx += xx
                err += dx
                     
    
        # /* plot remaining part to end */
        #plotLine(x0,y0, x2,y2)
        #return grid
    
    def RastCurve(points):
        return points
    
    # plotting the curve
    def PlotBezier(curve, controls, ctrl=False, save=True):
        plt.xticks([i1 for i1 in range(0, c.GRID_SIZE+1)]), plt.yticks([i1 for i1 in range(0, c.GRID_SIZE+1)])
        plt.xlim(0, c.GRID_SIZE)
        plt.ylim(0, c.GRID_SIZE)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().axes.yaxis.set_ticklabels([])
        plt.gca().axes.xaxis.set_ticklabels([])
        plt.grid(b=True, which='major', axis='both')
        
        plt.plot(curve[:, 0], curve[:, 1], color='blue')
        if ctrl:
            plt.plot(controls[:, 0], controls[:, 1], 'ro:', alpha=0.3)
        if save:
            plt.savefig("bezPlot.jpg", dpi=200)
        plt.show()
        
    # plotting the grid
    def PlotGrid(grid, name, save=True):
        fig, ax = plt.subplots()
        plt.xlim(0, c.GRID_SIZE)
        plt.ylim(0, c.GRID_SIZE)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(b=True, which='major', axis='both')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        im = ax.imshow(grid)#, cmap='Greys')
        if save:
            plt.savefig(name, dpi=200)
        plt.show()