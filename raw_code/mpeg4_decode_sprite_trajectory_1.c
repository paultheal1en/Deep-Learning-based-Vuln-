static void mpeg4_decode_sprite_trajectory(MpegEncContext * s, GetBitContext *gb)

{

    int i;

    int a= 2<<s->sprite_warping_accuracy;

    int rho= 3-s->sprite_warping_accuracy;

    int r=16/a;

    const int vop_ref[4][2]= {{0,0}, {s->width,0}, {0, s->height}, {s->width, s->height}}; // only true for rectangle shapes

    int d[4][2]={{0,0}, {0,0}, {0,0}, {0,0}};

    int sprite_ref[4][2];

    int virtual_ref[2][2];

    int w2, h2, w3, h3;

    int alpha=0, beta=0;

    int w= s->width;

    int h= s->height;

    int min_ab;



    for(i=0; i<s->num_sprite_warping_points; i++){

        int length;

        int x=0, y=0;



        length= get_vlc2(gb, sprite_trajectory.table, SPRITE_TRAJ_VLC_BITS, 3);

        if(length){

            x= get_xbits(gb, length);

        }

        if(!(s->divx_version==500 && s->divx_build==413)) skip_bits1(gb); /* marker bit */



        length= get_vlc2(gb, sprite_trajectory.table, SPRITE_TRAJ_VLC_BITS, 3);

        if(length){

            y=get_xbits(gb, length);

        }

        skip_bits1(gb); /* marker bit */

        s->sprite_traj[i][0]= d[i][0]= x;

        s->sprite_traj[i][1]= d[i][1]= y;

    }

    for(; i<4; i++)

        s->sprite_traj[i][0]= s->sprite_traj[i][1]= 0;



    while((1<<alpha)<w) alpha++;

    while((1<<beta )<h) beta++; // there seems to be a typo in the mpeg4 std for the definition of w' and h'

    w2= 1<<alpha;

    h2= 1<<beta;



// Note, the 4th point isn't used for GMC

    if(s->divx_version==500 && s->divx_build==413){

        sprite_ref[0][0]= a*vop_ref[0][0] + d[0][0];

        sprite_ref[0][1]= a*vop_ref[0][1] + d[0][1];

        sprite_ref[1][0]= a*vop_ref[1][0] + d[0][0] + d[1][0];

        sprite_ref[1][1]= a*vop_ref[1][1] + d[0][1] + d[1][1];

        sprite_ref[2][0]= a*vop_ref[2][0] + d[0][0] + d[2][0];

        sprite_ref[2][1]= a*vop_ref[2][1] + d[0][1] + d[2][1];

    } else {

        sprite_ref[0][0]= (a>>1)*(2*vop_ref[0][0] + d[0][0]);

        sprite_ref[0][1]= (a>>1)*(2*vop_ref[0][1] + d[0][1]);

        sprite_ref[1][0]= (a>>1)*(2*vop_ref[1][0] + d[0][0] + d[1][0]);

        sprite_ref[1][1]= (a>>1)*(2*vop_ref[1][1] + d[0][1] + d[1][1]);

        sprite_ref[2][0]= (a>>1)*(2*vop_ref[2][0] + d[0][0] + d[2][0]);

        sprite_ref[2][1]= (a>>1)*(2*vop_ref[2][1] + d[0][1] + d[2][1]);

    }

/*    sprite_ref[3][0]= (a>>1)*(2*vop_ref[3][0] + d[0][0] + d[1][0] + d[2][0] + d[3][0]);

    sprite_ref[3][1]= (a>>1)*(2*vop_ref[3][1] + d[0][1] + d[1][1] + d[2][1] + d[3][1]); */



// this is mostly identical to the mpeg4 std (and is totally unreadable because of that ...)

// perhaps it should be reordered to be more readable ...

// the idea behind this virtual_ref mess is to be able to use shifts later per pixel instead of divides

// so the distance between points is converted from w&h based to w2&h2 based which are of the 2^x form

    virtual_ref[0][0]= 16*(vop_ref[0][0] + w2)

        + ROUNDED_DIV(((w - w2)*(r*sprite_ref[0][0] - 16*vop_ref[0][0]) + w2*(r*sprite_ref[1][0] - 16*vop_ref[1][0])),w);

    virtual_ref[0][1]= 16*vop_ref[0][1]

        + ROUNDED_DIV(((w - w2)*(r*sprite_ref[0][1] - 16*vop_ref[0][1]) + w2*(r*sprite_ref[1][1] - 16*vop_ref[1][1])),w);

    virtual_ref[1][0]= 16*vop_ref[0][0]

        + ROUNDED_DIV(((h - h2)*(r*sprite_ref[0][0] - 16*vop_ref[0][0]) + h2*(r*sprite_ref[2][0] - 16*vop_ref[2][0])),h);

    virtual_ref[1][1]= 16*(vop_ref[0][1] + h2)

        + ROUNDED_DIV(((h - h2)*(r*sprite_ref[0][1] - 16*vop_ref[0][1]) + h2*(r*sprite_ref[2][1] - 16*vop_ref[2][1])),h);



    switch(s->num_sprite_warping_points)

    {

        case 0:

            s->sprite_offset[0][0]= 0;

            s->sprite_offset[0][1]= 0;

            s->sprite_offset[1][0]= 0;

            s->sprite_offset[1][1]= 0;

            s->sprite_delta[0][0]= a;

            s->sprite_delta[0][1]= 0;

            s->sprite_delta[1][0]= 0;

            s->sprite_delta[1][1]= a;

            s->sprite_shift[0]= 0;

            s->sprite_shift[1]= 0;

            break;

        case 1: //GMC only

            s->sprite_offset[0][0]= sprite_ref[0][0] - a*vop_ref[0][0];

            s->sprite_offset[0][1]= sprite_ref[0][1] - a*vop_ref[0][1];

            s->sprite_offset[1][0]= ((sprite_ref[0][0]>>1)|(sprite_ref[0][0]&1)) - a*(vop_ref[0][0]/2);

            s->sprite_offset[1][1]= ((sprite_ref[0][1]>>1)|(sprite_ref[0][1]&1)) - a*(vop_ref[0][1]/2);

            s->sprite_delta[0][0]= a;

            s->sprite_delta[0][1]= 0;

            s->sprite_delta[1][0]= 0;

            s->sprite_delta[1][1]= a;

            s->sprite_shift[0]= 0;

            s->sprite_shift[1]= 0;

            break;

        case 2:

            s->sprite_offset[0][0]= (sprite_ref[0][0]<<(alpha+rho))

                                                  + (-r*sprite_ref[0][0] + virtual_ref[0][0])*(-vop_ref[0][0])

                                                  + ( r*sprite_ref[0][1] - virtual_ref[0][1])*(-vop_ref[0][1])

                                                  + (1<<(alpha+rho-1));

            s->sprite_offset[0][1]= (sprite_ref[0][1]<<(alpha+rho))

                                                  + (-r*sprite_ref[0][1] + virtual_ref[0][1])*(-vop_ref[0][0])

                                                  + (-r*sprite_ref[0][0] + virtual_ref[0][0])*(-vop_ref[0][1])

                                                  + (1<<(alpha+rho-1));

            s->sprite_offset[1][0]= ( (-r*sprite_ref[0][0] + virtual_ref[0][0])*(-2*vop_ref[0][0] + 1)

                                     +( r*sprite_ref[0][1] - virtual_ref[0][1])*(-2*vop_ref[0][1] + 1)

                                     +2*w2*r*sprite_ref[0][0]

                                     - 16*w2

                                     + (1<<(alpha+rho+1)));

            s->sprite_offset[1][1]= ( (-r*sprite_ref[0][1] + virtual_ref[0][1])*(-2*vop_ref[0][0] + 1)

                                     +(-r*sprite_ref[0][0] + virtual_ref[0][0])*(-2*vop_ref[0][1] + 1)

                                     +2*w2*r*sprite_ref[0][1]

                                     - 16*w2

                                     + (1<<(alpha+rho+1)));

            s->sprite_delta[0][0]=   (-r*sprite_ref[0][0] + virtual_ref[0][0]);

            s->sprite_delta[0][1]=   (+r*sprite_ref[0][1] - virtual_ref[0][1]);

            s->sprite_delta[1][0]=   (-r*sprite_ref[0][1] + virtual_ref[0][1]);

            s->sprite_delta[1][1]=   (-r*sprite_ref[0][0] + virtual_ref[0][0]);



            s->sprite_shift[0]= alpha+rho;

            s->sprite_shift[1]= alpha+rho+2;

            break;

        case 3:

            min_ab= FFMIN(alpha, beta);

            w3= w2>>min_ab;

            h3= h2>>min_ab;

            s->sprite_offset[0][0]=  (sprite_ref[0][0]<<(alpha+beta+rho-min_ab))

                                   + (-r*sprite_ref[0][0] + virtual_ref[0][0])*h3*(-vop_ref[0][0])

                                   + (-r*sprite_ref[0][0] + virtual_ref[1][0])*w3*(-vop_ref[0][1])

                                   + (1<<(alpha+beta+rho-min_ab-1));

            s->sprite_offset[0][1]=  (sprite_ref[0][1]<<(alpha+beta+rho-min_ab))

                                   + (-r*sprite_ref[0][1] + virtual_ref[0][1])*h3*(-vop_ref[0][0])

                                   + (-r*sprite_ref[0][1] + virtual_ref[1][1])*w3*(-vop_ref[0][1])

                                   + (1<<(alpha+beta+rho-min_ab-1));

            s->sprite_offset[1][0]=  (-r*sprite_ref[0][0] + virtual_ref[0][0])*h3*(-2*vop_ref[0][0] + 1)

                                   + (-r*sprite_ref[0][0] + virtual_ref[1][0])*w3*(-2*vop_ref[0][1] + 1)

                                   + 2*w2*h3*r*sprite_ref[0][0]

                                   - 16*w2*h3

                                   + (1<<(alpha+beta+rho-min_ab+1));

            s->sprite_offset[1][1]=  (-r*sprite_ref[0][1] + virtual_ref[0][1])*h3*(-2*vop_ref[0][0] + 1)

                                   + (-r*sprite_ref[0][1] + virtual_ref[1][1])*w3*(-2*vop_ref[0][1] + 1)

                                   + 2*w2*h3*r*sprite_ref[0][1]

                                   - 16*w2*h3

                                   + (1<<(alpha+beta+rho-min_ab+1));

            s->sprite_delta[0][0]=   (-r*sprite_ref[0][0] + virtual_ref[0][0])*h3;

            s->sprite_delta[0][1]=   (-r*sprite_ref[0][0] + virtual_ref[1][0])*w3;

            s->sprite_delta[1][0]=   (-r*sprite_ref[0][1] + virtual_ref[0][1])*h3;

            s->sprite_delta[1][1]=   (-r*sprite_ref[0][1] + virtual_ref[1][1])*w3;



            s->sprite_shift[0]= alpha + beta + rho - min_ab;

            s->sprite_shift[1]= alpha + beta + rho - min_ab + 2;

            break;

    }

    /* try to simplify the situation */

    if(   s->sprite_delta[0][0] == a<<s->sprite_shift[0]

       && s->sprite_delta[0][1] == 0

       && s->sprite_delta[1][0] == 0

       && s->sprite_delta[1][1] == a<<s->sprite_shift[0])

    {

        s->sprite_offset[0][0]>>=s->sprite_shift[0];

        s->sprite_offset[0][1]>>=s->sprite_shift[0];

        s->sprite_offset[1][0]>>=s->sprite_shift[1];

        s->sprite_offset[1][1]>>=s->sprite_shift[1];

        s->sprite_delta[0][0]= a;

        s->sprite_delta[0][1]= 0;

        s->sprite_delta[1][0]= 0;

        s->sprite_delta[1][1]= a;

        s->sprite_shift[0]= 0;

        s->sprite_shift[1]= 0;

        s->real_sprite_warping_points=1;

    }

    else{

        int shift_y= 16 - s->sprite_shift[0];

        int shift_c= 16 - s->sprite_shift[1];

        for(i=0; i<2; i++){

            s->sprite_offset[0][i]<<= shift_y;

            s->sprite_offset[1][i]<<= shift_c;

            s->sprite_delta[0][i]<<= shift_y;

            s->sprite_delta[1][i]<<= shift_y;

            s->sprite_shift[i]= 16;

        }

        s->real_sprite_warping_points= s->num_sprite_warping_points;

    }

}
