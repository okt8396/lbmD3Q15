#ifndef _LBMD2Q9_MPI_HPP_
#define _LBMD2Q9_MPI_HPP_

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <mpi.h>

// Helper class for creating barriers
class Barrier
{
    public:
        enum Type {HORIZONTAL, VERTICAL};
    protected:
        Type type;
        int x1;
        int x2;
        int y1;
        int y2;

    public:
        Type getType() { return type; }
        int getX1() { return x1; }
        int getX2() { return x2; }
        int getY1() { return y1; }
        int getY2() { return y2; }
};

class BarrierHorizontal : public Barrier
{
    public:
        BarrierHorizontal(int x_start, int x_end, int y) {
            type = Barrier::HORIZONTAL;
            x1 = x_start;
            x2 = x_end;
            y1 = y;
            y2 = y;
        }
        ~BarrierHorizontal() {}
};

class BarrierVertical : public Barrier
{
    public:
        BarrierVertical(int y_start, int y_end, int x) {
            type = Barrier::VERTICAL;
            x1 = x;
            x2 = x;
            y1 = y_start;
            y2 = y_end;
        }
        ~BarrierVertical() {}
};

// Lattice-Boltzman Methods CFD simulation
class LbmD2Q9
{
    public:
        enum FluidProperty {None, Density, Speed, Vorticity};

    private:
        enum Neighbor {NeighborN, NeighborE, NeighborS, NeighborW, NeighborNE, NeighborNW, NeighborSE, NeighborSW};
        enum Column {LeftBoundaryCol, LeftCol, RightCol, RightBoundaryCol};

        int rank;
        int num_ranks;
        uint32_t total_x;
        uint32_t total_y;
        uint32_t dim_x;
        uint32_t dim_y;
        uint32_t start_x;
        uint32_t start_y;
        uint32_t num_x;
        uint32_t num_y;
        int offset_x;
        int offset_y;
        //uint32_t *rank_local_size;
        //uint32_t *rank_local_start;
        double speed_scale;
        double *f_0;
        double *f_1;
        double *f_2;
        double *f_3;
        double *f_4;
        double *f_5;
        double *f_6;
        double *f_7;
        double *f_8;
	double *f_9;
	double *f_10;
	double *f_11;
	double *f_12;
	double *f_13;
	double *f_14;
	double *density;
        double *velocity_x;
        double *velocity_y;
        double *vorticity;
        double *speed;
        bool *barrier;
        FluidProperty stored_property;
        double *recv_buf;
        bool *brecv_buf;
        int neighbors[8];
        MPI_Datatype columns_2d[4];
        MPI_Datatype own_scalar;
        MPI_Datatype own_bool;
        MPI_Datatype *other_scalar;
        MPI_Datatype *other_bool;

        void setEquilibrium(int x, int y, double new_velocity_x, double new_velocity_y, double new_density);
        void getClosestFactors2(int value, int *factor_1, int *factor_2);
        void exchangeBoundaries();

    public:
        LbmD2Q9(uint32_t width, uint32_t height, double scale, int task_id, int num_tasks);
        ~LbmD2Q9();

        void initBarrier(std::vector<Barrier*> barriers);
        void initFluid(double physical_speed);
        void updateFluid(double physical_speed);
        void collide(double viscosity);
        void stream();
        void bounceBackStream();
        bool checkStability();
        void computeSpeed();
        void computeVorticity();
        void gatherDataOnRank0(FluidProperty property);
        uint32_t getDimX();
        uint32_t getDimY();
        uint32_t getTotalDimX();
        uint32_t getTotalDimY();
        uint32_t getOffsetX();
        uint32_t getOffsetY();
        uint32_t getStartX();
        uint32_t getStartY();
        uint32_t getSizeX();
        uint32_t getSizeY();
        //uint32_t* getRankLocalSize(int rank);
        //uint32_t* getRankLocalStart(int rank);
        bool* getBarrier();
        double* getDensity();
        double* getVelocityX();
        double* getVelocityY();
        double* getVorticity();
        double* getSpeed();
};

// constructor
LbmD2Q9::LbmD2Q9(uint32_t width, uint32_t height, double scale, int task_id, int num_tasks)
{
    rank = task_id;
    num_ranks = num_tasks;
    speed_scale = scale;
    stored_property = None;

    // split up problem space
    int n_x, n_y, col, row, chunk_w, chunk_h, extra_w, extra_h;
    int neighbor_cols, neighbor_rows;
    getClosestFactors2(num_ranks, &n_x, &n_y);
    chunk_w = width / n_x;
    chunk_h = height / n_y;
    extra_w = width % n_x;
    extra_h = height % n_y;
    col = rank % n_x;
    row = rank / n_x;
    num_x = chunk_w + ((col < extra_w) ? 1 : 0);
    num_y = chunk_h + ((row < extra_h) ? 1 : 0);
    offset_x = col * chunk_w + std::min(col, extra_w);
    offset_y = row * chunk_h + std::min(row, extra_h);
    neighbor_cols = (num_ranks == 1) ? 0 : ((col == 0 || col == n_x-1) ? 1 : 2);
    neighbor_rows = (num_ranks == 1) ? 0 : ((row == 0 || row == n_y-1) ? 1 : 2);
    start_x = (col == 0) ? 0 : 1;
    start_y = (row == 0) ? 0 : 1;
    neighbors[NeighborN] = (row == n_y-1) ? -1 : rank + n_x;
    neighbors[NeighborE] = (col == n_x-1) ? -1 : rank + 1;
    neighbors[NeighborS] = (row == 0) ? -1 : rank - n_x; 
    neighbors[NeighborW] = (col == 0) ? -1 : rank - 1;
    neighbors[NeighborNE] = (row == n_y-1 || col == n_x-1) ? -1 : rank + n_x + 1;
    neighbors[NeighborNW] = (row == n_y-1 || col == 0) ? -1 : rank + n_x - 1;
    neighbors[NeighborSE] = (row == 0 || col == n_x-1) ? -1 : rank - n_x + 1;
    neighbors[NeighborSW] = (row == 0 || col == 0) ? -1: rank - n_x - 1;
    // create data types for exchanging data with neighbors
    int block_width, block_height, array[2], subsize[2], offsets[2];
    block_width = num_x + neighbor_cols;
    block_height = num_y + neighbor_rows;
    array[0] = block_height;
    array[1] = block_width;
    subsize[0] = num_y;
    subsize[1] = 1;
    offsets[0] = start_y;
    offsets[1] = 0;
    MPI_Type_create_subarray(2, array, subsize, offsets, MPI_ORDER_C, MPI_DOUBLE, &columns_2d[LeftBoundaryCol]);
    MPI_Type_commit(&columns_2d[LeftBoundaryCol]); // left boundary column
    offsets[1] = start_x;
    MPI_Type_create_subarray(2, array, subsize, offsets, MPI_ORDER_C, MPI_DOUBLE, &columns_2d[LeftCol]);
    MPI_Type_commit(&columns_2d[LeftCol]); // left column
    offsets[1] = start_x + num_x - 1;
    MPI_Type_create_subarray(2, array, subsize, offsets, MPI_ORDER_C, MPI_DOUBLE, &columns_2d[RightCol]);
    MPI_Type_commit(&columns_2d[RightCol]); // right column
    offsets[1] = block_width - 1;
    MPI_Type_create_subarray(2, array, subsize, offsets, MPI_ORDER_C, MPI_DOUBLE, &columns_2d[RightBoundaryCol]);
    MPI_Type_commit(&columns_2d[RightBoundaryCol]); // right boundary column
    // create data types for gathering data 
    subsize[1] = num_x;
    offsets[1] = start_x;
    MPI_Type_create_subarray(2, array, subsize, offsets, MPI_ORDER_C, MPI_DOUBLE, &own_scalar);
    MPI_Type_commit(&own_scalar);
    MPI_Type_create_subarray(2, array, subsize, offsets, MPI_ORDER_C, MPI_BYTE, &own_bool);
    MPI_Type_commit(&own_bool);
    other_scalar = new MPI_Datatype[num_ranks];
    other_bool = new MPI_Datatype[num_ranks];
    int i, other_col, other_row;
    array[0] = height;
    array[1] = width;
    //rank_local_size = new uint32_t[2 * num_ranks];
    //rank_local_start = new uint32_t[2 * num_ranks];
    for (i=0; i<num_ranks; i++)
    {
        other_col = i % n_x;
        other_row = i / n_x;
        subsize[0] = chunk_h + ((other_row < extra_h) ? 1 : 0);
        subsize[1] = chunk_w + ((other_col < extra_w) ? 1 : 0);
        offsets[0] = other_row * chunk_h + std::min(other_row, extra_h);
        offsets[1] = other_col * chunk_w + std::min(other_col, extra_w);
        //rank_local_size[2 * i + 0] = subsize[1];
        //rank_local_size[2 * i + 1] = subsize[0];
        //rank_local_start[2 * i + 0] = (other_col == 0) ? 0 : 1;
        //rank_local_start[2 * i + 1] = (other_row == 0) ? 0 : 1;
        MPI_Type_create_subarray(2, array, subsize, offsets, MPI_ORDER_C, MPI_DOUBLE, &other_scalar[i]);
        MPI_Type_commit(&other_scalar[i]);
        MPI_Type_create_subarray(2, array, subsize, offsets, MPI_ORDER_C, MPI_BYTE, &other_bool[i]);
        MPI_Type_commit(&other_bool[i]);
    }

    // set up sub grid for simulation
    total_x = width;
    total_y = height;
    dim_x = block_width;
    dim_y = block_height;
    
    recv_buf = new double[total_x * total_y];
    brecv_buf = new bool[total_x * total_y];
    
    uint32_t size = dim_x * dim_y;

    // allocate all double arrays at once
    double *dbl_arrays = new double[14 * size];

    // set array pointers
    f_0        = dbl_arrays;
    f_1        = dbl_arrays + (   size);
    f_2        = dbl_arrays + ( 2*size);
    f_3        = dbl_arrays + ( 3*size);
    f_4        = dbl_arrays + ( 4*size);
    f_5       = dbl_arrays + ( 5*size);
    f_6       = dbl_arrays + ( 6*size);
    f_7       = dbl_arrays + ( 7*size);
    f_8       = dbl_arrays + ( 8*size);
    density    = dbl_arrays + ( 9*size);
    velocity_x = dbl_arrays + (10*size);
    velocity_y = dbl_arrays + (11*size);
    vorticity  = dbl_arrays + (12*size);
    speed      = dbl_arrays + (13*size);
    
    // allocate boolean array
    barrier = new bool[size];
}

// destructor
LbmD2Q9::~LbmD2Q9()
{
    MPI_Type_free(&columns_2d[LeftBoundaryCol]);
    MPI_Type_free(&columns_2d[LeftCol]);
    MPI_Type_free(&columns_2d[RightCol]);
    MPI_Type_free(&columns_2d[RightBoundaryCol]);

    MPI_Type_free(&own_scalar);
    MPI_Type_free(&own_bool);
    int i;
    for (i=0; i<num_ranks; i++)
    {
        MPI_Type_free(&other_scalar[i]);
        MPI_Type_free(&other_bool[i]);
    }

    //delete[] rank_local_size;
    //delete[] rank_local_start;
    delete[] f_0;
    delete[] barrier;
}



// initialize barrier based on selected type
void LbmD2Q9::initBarrier(std::vector<Barrier*> barriers)
{
    
    // clear barrier to all `false`
    memset(barrier, 0, dim_x * dim_y);
    
    // set barrier to `true` where horizontal or vertical barriers exist
    int sx = (offset_x == 0) ? 0 : offset_x - 1;
    int sy = (offset_y == 0) ? 0 : offset_y - 1;
    int i, j;
    for (i = 0; i < barriers.size(); i++) {
        if (barriers[i]->getType() == Barrier::Type::HORIZONTAL) {
            int y = barriers[i]->getY1() - sy;
            if (y >= 0 && y < dim_y) {
                for (j = barriers[i]->getX1(); j <= barriers[i]->getX2(); j++) {
                    int x = j - sx;
                    if (x >= 0 && x < dim_x) {
                        barrier[y * dim_x + x] = true;
                    }
                }
            }
        }
        else {                // Barrier::VERTICAL
            int x = barriers[i]->getX1() - sx;
            if (x >= 0 && x < dim_x) {
                for (j = barriers[i]->getY1(); j <= barriers[i]->getY2(); j++) {
                    int y = j - sy;
                    if (y >= 0 && y < dim_y) {
                        barrier[y * dim_x + x] = true;
                    }
                }
            }
        }
    }
}

// initialize fluid
void LbmD2Q9::initFluid(double physical_speed)
{
    int i, j, row;
    double speed = speed_scale * physical_speed;
    for (j = 0; j < dim_y; j++)
    {
        row = j * dim_x;
        for (i = 0; i < dim_x; i++)
        {
            setEquilibrium(i, j, speed, 0.0, 1.0);
            vorticity[row + i] = 0.0;
        }
    }
}

void LbmD2Q9::updateFluid(double physical_speed)
{
    int i;
    double speed = speed_scale * physical_speed;
    for (i = 0; i < dim_x; i++)
    {
        setEquilibrium(i, 0, speed, 0.0, 1.0);
        setEquilibrium(i, dim_y - 1, speed, 0.0, 1.0);
    }
    for (i = 1; i < dim_y - 1; i++)
    {
        setEquilibrium(0, i, speed, 0.0, 1.0);
        setEquilibrium(dim_x - 1, i, speed, 0.0, 1.0);
    }
}

// particle collision
void LbmD2Q9::collide(double viscosity)
{
    int i, j, row, idx;
    double omega = 1.0 / (3.0 * viscosity + 0.5); // reciprocal of relaxation time
    for (j = 1; j < dim_y - 1; j++)
    {
        row = j * dim_x;
        for (i = 1; i < dim_x - 1; i++)
        {
            idx = row + i;
            density[idx] = f_0[idx] + f_1[idx] + f_3[idx] + f_2[idx] + f_4[idx] + f_6[idx] + f_5[idx] + f_8[idx] + f_7[idx];
            velocity_x[idx] = (f_2[idx] + f_5[idx] + f_7[idx] - f_4[idx] - f_6[idx] - f_8[idx]) / density[idx];
            velocity_y[idx] = (f_1[idx] + f_5[idx] + f_6[idx] - f_3[idx] - f_7[idx] - f_8[idx]) / density[idx];
            double one_ninth_density       = (1.0 /  9.0) * density[idx];
            double four_ninths_density     = (4.0 /  9.0) * density[idx];
            double one_thirtysixth_density = (1.0 / 36.0) * density[idx];
            double velocity_3x   = 3.0 * velocity_x[idx];
            double velocity_3y   = 3.0 * velocity_y[idx];
            double velocity_x2   = velocity_x[idx] * velocity_x[idx];
            double velocity_y2   = velocity_y[idx] * velocity_y[idx];
            double velocity_2xy  = 2.0 * velocity_x[idx] * velocity_y[idx];
            double vecocity_2    = velocity_x2 + velocity_y2;
            double vecocity_2_15 = 1.5 * vecocity_2;
            f_0[idx]  += omega * (four_ninths_density     * (1                                                                 - vecocity_2_15) - f_0[idx]);
            f_2[idx]  += omega * (one_ninth_density       * (1 + velocity_3x               + 4.5 * velocity_x2                 - vecocity_2_15) - f_2[idx]);
            f_4[idx]  += omega * (one_ninth_density       * (1 - velocity_3x               + 4.5 * velocity_x2                 - vecocity_2_15) - f_4[idx]);
            f_1[idx]  += omega * (one_ninth_density       * (1 + velocity_3y               + 4.5 * velocity_y2                 - vecocity_2_15) - f_1[idx]);
            f_3[idx]  += omega * (one_ninth_density       * (1 - velocity_3y               + 4.5 * velocity_y2                 - vecocity_2_15) - f_3[idx]);
            f_5[idx] += omega * (one_thirtysixth_density * (1 + velocity_3x + velocity_3y + 4.5 * (vecocity_2 + velocity_2xy) - vecocity_2_15) - f_5[idx]);
            f_7[idx] += omega * (one_thirtysixth_density * (1 + velocity_3x - velocity_3y + 4.5 * (vecocity_2 - velocity_2xy) - vecocity_2_15) - f_7[idx]);
            f_6[idx] += omega * (one_thirtysixth_density * (1 - velocity_3x + velocity_3y + 4.5 * (vecocity_2 - velocity_2xy) - vecocity_2_15) - f_6[idx]);
            f_8[idx] += omega * (one_thirtysixth_density * (1 - velocity_3x - velocity_3y + 4.5 * (vecocity_2 + velocity_2xy) - vecocity_2_15) - f_8[idx]);
        }
    }

    exchangeBoundaries();
}

// particle streaming
void LbmD2Q9::stream()
{
    int i, j, row, rowp, rown, idx;
    for (j = dim_y - 2; j > 0; j--) // first start in NW corner...
    {
        row = j * dim_x;
        rowp = (j - 1) * dim_x;
        for (i = 1; i < dim_x - 1; i++)
        {
            f_1[row + i] =  f_1[rowp + i];
            f_6[row + i] = f_6[rowp + i + 1];
        }
    }
    for (j = dim_y - 2; j > 0; j--) // then start in NE corner...
    {
        row = j * dim_x;
        rowp = (j - 1) * dim_x;
        for (i = dim_x - 2; i > 0; i--)
        {
            f_2[row + i] =  f_2[row + i - 1];
            f_5[row + i] = f_5[rowp + i - 1];
        }
    }
    for (j = 1; j < dim_y - 1; j++) // then start in SE corner...
    {
        row = j * dim_x;
        rown = (j + 1) * dim_x;
        for (i = dim_x - 2; i > 0; i--)
        {
            f_3[row + i] =  f_3[rown + i];
            f_7[row + i] = f_7[rown + i - 1];
        }
    }
    for (j = 1; j < dim_y - 1; j++) // then start in the SW corner...
    {
        row = j * dim_x;
        rown = (j + 1) * dim_x;
        for (i = 1; i < dim_x - 1; i++)
        {
            f_4[row + i] =  f_4[row + i + 1];
            f_8[row + i] = f_8[rown + i + 1];
        }
    }

    exchangeBoundaries();
}

// particle streaming bouncing back off of barriers
void LbmD2Q9::bounceBackStream()
{
    int i, j, row, rowp, rown, idx;
    for (j = 1; j < dim_y - 1; j++) // handle bounce-back from barriers
    {
        row = j * dim_x;
        rowp = (j - 1) * dim_x;
        rown = (j + 1) * dim_x;
        for (i = 1; i < dim_x - 1; i++)
        {
            idx = row + i;
            if (barrier[row + i - 1])
            {
                f_2[idx] = f_4[row + i - 1];
            }
            if (barrier[row + i + 1])
            {
                f_4[idx] = f_2[row + i + 1];
            }
            if (barrier[rowp + i])
            {
                f_1[idx] =   f_3[rowp + i];
            }
            if (barrier[rown + i])
            {
                f_3[idx] =   f_1[rown + i];
            }
            if (barrier[rowp + i - 1])
            {
                f_5[idx] =  f_8[rowp + i - 1];
            }
            if (barrier[rowp + i + 1])
            {
                f_6[idx] =  f_7[rowp + i + 1];
            }
            if (barrier[rown + i - 1])
            {
                f_7[idx] =  f_6[rown + i - 1];
            }
            if (barrier[rown + i + 1])
            {
                f_8[idx] =  f_5[rown + i + 1];
            }
        }
    }
}

// check if simulation has become unstable (if so, more time steps are required)
bool LbmD2Q9::checkStability()
{
    int i, idx;
    bool stable = true;

    for (i = 0; i < dim_x; i++)
    {
        idx = (dim_y / 2) * dim_x + i;
        if (density[idx] <= 0)
        {
            stable = false;
        }
    }
    return stable;
}

// compute speed (magnitude of velocity vector)
void LbmD2Q9::computeSpeed()
{
    int i, j, row;
    for (j = 1; j < dim_y - 1; j++)
    {
        row = j * dim_x;
        for (i = 1; i < dim_x - 1; i++)
        {
            speed[row + i] = sqrt(velocity_x[row + i] * velocity_x[row + i] + velocity_y[row + i] * velocity_y[row + i]);
        }
    }
}

// compute vorticity (rotational velocity)
void LbmD2Q9::computeVorticity()
{
    int i, j, row, rowp, rown;
    for (j = 1; j < dim_y - 1; j++)
    {
        row = j * dim_x;
        rowp = (j - 1) * dim_x;
        rown = (j + 1) * dim_x;
        for (i = 1; i < dim_x - 1; i++)
        {
            vorticity[row + i] = velocity_y[row + i + 1] - velocity_y[row + i - 1] - velocity_x[rown + i] + velocity_x[rowp + i];
        }
    }
}

// gather all data on rank 0
void LbmD2Q9::gatherDataOnRank0(FluidProperty property)
{
    double *send_buf = NULL;
    bool *bsend_buf = barrier;
    switch (property)
    {
        case Density:
            send_buf = density;
            break;
        case Speed:
            send_buf = speed;
            break;
        case Vorticity:
            send_buf = vorticity;
            break;
        case None:
            return;
    }

    MPI_Status status;
    MPI_Request request;
    if (rank == 0)
    {
        int i;
        MPI_Sendrecv(send_buf, 1, own_scalar, 0, 0, recv_buf, 1, other_scalar[0], 0, 0, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(bsend_buf, 1, own_bool, 0, 0, brecv_buf, 1, other_bool[0], 0, 0, MPI_COMM_WORLD, &status);
        for (i = 1; i < num_ranks; i++)
        {
            MPI_Recv(recv_buf, 1, other_scalar[i], i, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(brecv_buf, 1, other_bool[i], i, 1, MPI_COMM_WORLD, &status);
        }
    }
    else
    {
        MPI_Send(send_buf, 1, own_scalar, 0, 0, MPI_COMM_WORLD);
        MPI_Send(bsend_buf, 1, own_bool, 0, 1, MPI_COMM_WORLD);
    }

    stored_property = property;
}

// get width of sub-area this task owns (including ghost cells)
uint32_t LbmD2Q9::getDimX()
{
    return dim_x;
}

// get width of sub-area this task owns (including ghost cells)
uint32_t LbmD2Q9::getDimY()
{
    return dim_y;
}

// get width of total area of simulation
uint32_t LbmD2Q9::getTotalDimX()
{
    return total_x;
}

// get width of total area of simulation
uint32_t LbmD2Q9::getTotalDimY()
{
    return total_y;
}

// get x offset into overall domain where this sub-area esxists
uint32_t LbmD2Q9::getOffsetX()
{
    return offset_x;
}

// get y offset into overall domain where this sub-area esxists
uint32_t LbmD2Q9::getOffsetY()
{
    return offset_y;
}

// get x start for valid data (0 if no ghost cell on left, 1 if there is a ghost cell on left)
uint32_t LbmD2Q9::getStartX()
{
    return start_x;
}

// get y start for valid data (0 if no ghost cell on top, 1 if there is a ghost cell on top)
uint32_t LbmD2Q9::getStartY()
{
    return start_y;
}

// get width of sub-area this task is responsible for (excluding ghost cells)
uint32_t LbmD2Q9::getSizeX()
{
    return num_x;
}

// get width of sub-area this task is responsible for (excluding ghost cells)
uint32_t LbmD2Q9::getSizeY()
{
    return num_y;
}

// get the local width and height of a particular rank's data
//uint32_t* LbmD2Q9::getRankLocalSize(int rank)
//{
//    return rank_local_size + (2 * rank);
//}

// get the local x and y start of a particular rank's data
//uint32_t* LbmD2Q9::getRankLocalStart(int rank)
//{
//    return rank_local_start + (2 * rank);
//}

// get barrier array
bool* LbmD2Q9::getBarrier()
{
    if (rank != 0) return NULL;
    return brecv_buf;
}


// get density array
double* LbmD2Q9::getDensity()
{
    return density;
}

// get velocity x array
double* LbmD2Q9::getVelocityX()
{
    return velocity_x;
}

// get density array
double* LbmD2Q9::getVelocityY()
{
    return velocity_y;
}

// get vorticity array
double* LbmD2Q9::getVorticity()
{
    return vorticity;
}

// get vorticity array
double* LbmD2Q9::getSpeed()
{
    return speed;
}


/*
// get density array
double* LbmD2Q9::getDensity()
{
    if (rank != 0 || stored_property != Density) return NULL;
    return recv_buf;
}

// get vorticity array
double* LbmD2Q9::getVorticity()
{
    if (rank != 0 || stored_property != Vorticity) return NULL;
    return recv_buf;
}

// get speed array
double* LbmD2Q9::getSpeed()
{
    if (rank != 0 || stored_property != Speed) return NULL;
    return recv_buf;
}
*/

// private - set fluid equalibrium
void LbmD2Q9::setEquilibrium(int x, int y, double new_velocity_x, double new_velocity_y, double new_density)
{
    int idx = y * dim_x + x;

    double one_ninth = 1.0 / 9.0;
    double four_ninths = 4.0 / 9.0;
    double one_thirtysixth = 1.0 / 36.0;

    double velocity_3x   = 3.0 * new_velocity_x;
    double velocity_3y   = 3.0 * new_velocity_y;
    double velocity_x2   = new_velocity_x * new_velocity_x;
    double velocity_y2   = new_velocity_y * new_velocity_y;
    double velocity_2xy  = 2.0 * new_velocity_x * new_velocity_y;
    double vecocity_2    = velocity_x2 + velocity_y2;
    double vecocity_2_15 = 1.5 * vecocity_2;
    f_0[idx]  = four_ninths     * new_density * (1.0                                                                 - vecocity_2_15);
    f_2[idx]  = one_ninth       * new_density * (1.0 + velocity_3x               + 4.5 * velocity_x2                 - vecocity_2_15);
    f_4[idx]  = one_ninth       * new_density * (1.0 - velocity_3x               + 4.5 * velocity_x2                 - vecocity_2_15);
    f_1[idx]  = one_ninth       * new_density * (1.0 + velocity_3y               + 4.5 * velocity_y2                 - vecocity_2_15);
    f_3[idx]  = one_ninth       * new_density * (1.0 - velocity_3y               + 4.5 * velocity_y2                 - vecocity_2_15);
    f_5[idx] = one_thirtysixth * new_density * (1.0 + velocity_3x + velocity_3y + 4.5 * (vecocity_2 + velocity_2xy) - vecocity_2_15);
    f_7[idx] = one_thirtysixth * new_density * (1.0 + velocity_3x - velocity_3y + 4.5 * (vecocity_2 - velocity_2xy) - vecocity_2_15);
    f_6[idx] = one_thirtysixth * new_density * (1.0 - velocity_3x + velocity_3y + 4.5 * (vecocity_2 - velocity_2xy) - vecocity_2_15);
    f_8[idx] = one_thirtysixth * new_density * (1.0 - velocity_3x - velocity_3y + 4.5 * (vecocity_2 + velocity_2xy) - vecocity_2_15);
    density[idx]    = new_density;
    velocity_x[idx] = new_velocity_x;
    velocity_y[idx] = new_velocity_y;
}

// private - get 2 factors of a given number that are closest to each other
void LbmD2Q9::getClosestFactors2(int value, int *factor_1, int *factor_2)
{
    int test_num = (int)sqrt(value);
    while (value % test_num != 0)
    {
        test_num--;
    }
    *factor_2 = test_num;
    *factor_1 = value / test_num;
}

// private - exchange boundary information between MPI ranks
void LbmD2Q9::exchangeBoundaries()
{
    MPI_Status status;
    int nx = dim_x;
    int ny = dim_y;
    int sx = start_x;
    int sy = start_y;
    int cx = num_x;
    int cy = num_y;

    if (neighbors[NeighborN] >= 0)
    {
        MPI_Sendrecv(&(f_0[(ny - 2) * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_0[(ny - 1) * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_1[(ny - 2) * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_1[(ny - 1) * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_2[(ny - 2) * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_2[(ny - 1) * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_3[(ny - 2) * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_3[(ny - 1) * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_4[(ny - 2) * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_4[(ny - 1) * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_5[(ny - 2) * nx + sx]),       cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_5[(ny - 1) * nx + sx]),       cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_6[(ny - 2) * nx + sx]),       cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_6[(ny - 1) * nx + sx]),       cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_7[(ny - 2) * nx + sx]),       cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_7[(ny - 1) * nx + sx]),       cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_8[(ny - 2) * nx + sx]),       cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_8[(ny - 1) * nx + sx]),       cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(density[(ny - 2) * nx + sx]),    cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(density[(ny - 1) * nx + sx]),    cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(velocity_x[(ny - 2) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(velocity_x[(ny - 1) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(velocity_y[(ny - 2) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(velocity_y[(ny - 1) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
    }
    if (neighbors[NeighborE] >= 0)
    {
        MPI_Sendrecv(f_0,        1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_0,        1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_1,        1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_1,        1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_2,        1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_2,        1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_3,        1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_3,        1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_4,        1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_4,        1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_5,       1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_5,       1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_6,       1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_6,       1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_7,       1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_7,       1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_8,       1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_8,       1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(density,    1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, density,    1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(velocity_x, 1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, velocity_x, 1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(velocity_y, 1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, velocity_y, 1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
    }
    if (neighbors[NeighborS] >= 0)
    {
        MPI_Sendrecv(&(f_0[sy * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_0[sx]),        cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_1[sy * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_1[sx]),        cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_2[sy * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_2[sx]),        cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_3[sy * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_3[sx]),        cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_4[sy * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_4[sx]),        cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_5[sy * nx + sx]),       cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_5[sx]),       cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_6[sy * nx + sx]),       cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_6[sx]),       cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_7[sy * nx + sx]),       cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_7[sx]),       cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_8[sy * nx + sx]),       cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_8[sx]),       cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(density[sy * nx + sx]),    cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(density[sx]),    cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(velocity_x[sy * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(velocity_x[sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(velocity_y[sy * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(velocity_y[sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
    }
    if (neighbors[NeighborW] >= 0)
    {
        MPI_Sendrecv(f_0,        1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_0,        1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_1,        1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_1,        1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_2,        1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_2,        1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_3,        1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_3,        1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_4,        1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_4,        1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_5,       1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_5,       1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_6,       1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_6,       1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_7,       1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_7,       1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_8,       1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_8,       1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(density,    1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, density,    1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(velocity_x, 1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, velocity_x, 1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(velocity_y, 1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, velocity_y, 1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
    }
    if (neighbors[NeighborNE] >= 0)
    {
        MPI_Sendrecv(&(f_0[(ny - 2) * nx + nx - 2]),        1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_0[(ny - 1) * nx + nx - 1]),        1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_1[(ny - 2) * nx + nx - 2]),        1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_1[(ny - 1) * nx + nx - 1]),        1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_2[(ny - 2) * nx + nx - 2]),        1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_2[(ny - 1) * nx + nx - 1]),        1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_3[(ny - 2) * nx + nx - 2]),        1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_3[(ny - 1) * nx + nx - 1]),        1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_4[(ny - 2) * nx + nx - 2]),        1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_4[(ny - 1) * nx + nx - 1]),        1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_5[(ny - 2) * nx + nx - 2]),       1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_5[(ny - 1) * nx + nx - 1]),       1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_6[(ny - 2) * nx + nx - 2]),       1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_6[(ny - 1) * nx + nx - 1]),       1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_7[(ny - 2) * nx + nx - 2]),       1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_7[(ny - 1) * nx + nx - 1]),       1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_8[(ny - 2) * nx + nx - 2]),       1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_8[(ny - 1) * nx + nx - 1]),       1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(density[(ny - 2) * nx + nx - 2]),    1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(density[(ny - 1) * nx + nx - 1]),    1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(velocity_x[(ny - 2) * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(velocity_x[(ny - 1) * nx + nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(velocity_y[(ny - 2) * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(velocity_y[(ny - 1) * nx + nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
    }
    if (neighbors[NeighborNW] >= 0)
    {
        MPI_Sendrecv(&(f_0[(ny - 2) * nx + sx]),        1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_0[(ny - 1) * nx]),        1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_1[(ny - 2) * nx + sx]),        1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_1[(ny - 1) * nx]),        1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_2[(ny - 2) * nx + sx]),        1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_2[(ny - 1) * nx]),        1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_3[(ny - 2) * nx + sx]),        1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_3[(ny - 1) * nx]),        1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_4[(ny - 2) * nx + sx]),        1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_4[(ny - 1) * nx]),        1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_5[(ny - 2) * nx + sx]),       1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_5[(ny - 1) * nx]),       1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_6[(ny - 2) * nx + sx]),       1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_6[(ny - 1) * nx]),       1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_7[(ny - 2) * nx + sx]),       1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_7[(ny - 1) * nx]),       1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_8[(ny - 2) * nx + sx]),       1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_8[(ny - 1) * nx]),       1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(density[(ny - 2) * nx + sx]),    1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(density[(ny - 1) * nx]),    1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(velocity_x[(ny - 2) * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(velocity_x[(ny - 1) * nx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(velocity_y[(ny - 2) * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(velocity_y[(ny - 1) * nx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
    }
    if (neighbors[NeighborSE] >= 0)
    {
        MPI_Sendrecv(&(f_0[sy * nx + nx - 2]),        1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_0[nx - 1]),        1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_1[sy * nx + nx - 2]),        1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_1[nx - 1]),        1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_2[sy * nx + nx - 2]),        1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_2[nx - 1]),        1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_3[sy * nx + nx - 2]),        1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_3[nx - 1]),        1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_4[sy * nx + nx - 2]),        1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_4[nx - 1]),        1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_5[sy * nx + nx - 2]),       1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_5[nx - 1]),       1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_6[sy * nx + nx - 2]),       1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_6[nx - 1]),       1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_7[sy * nx + nx - 2]),       1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_7[nx - 1]),       1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_8[sy * nx + nx - 2]),       1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_8[nx - 1]),       1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(density[sy * nx + nx - 2]),    1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(density[nx - 1]),    1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(velocity_x[sy * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(velocity_x[nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(velocity_y[sy * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(velocity_y[nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
    }
    if (neighbors[NeighborSW] >= 0)
    {
        MPI_Sendrecv(&(f_0[sy * nx + sx]),        1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_0[0]),        1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_1[sy * nx + sx]),        1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_1[0]),        1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_2[sy * nx + sx]),        1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_2[0]),        1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_3[sy * nx + sx]),        1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_3[0]),        1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_4[sy * nx + sx]),        1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_4[0]),        1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_5[sy * nx + sx]),       1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_5[0]),       1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_6[sy * nx + sx]),       1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_6[0]),       1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_7[sy * nx + sx]),       1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_7[0]),       1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_8[sy * nx + sx]),       1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_8[0]),       1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(density[sy * nx + sx]),    1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(density[0]),    1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(velocity_x[sy * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(velocity_x[0]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(velocity_y[sy * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(velocity_y[0]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
    }
}


#endif // _LBMD2Q9_MPI_HPP_
