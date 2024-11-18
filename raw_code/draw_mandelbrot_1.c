static void draw_mandelbrot(AVFilterContext *ctx, uint32_t *color, int linesize, int64_t pts)

{

    MBContext *mb = ctx->priv;

    int x,y,i, in_cidx=0, next_cidx=0, tmp_cidx;

    double scale= mb->start_scale*pow(mb->end_scale/mb->start_scale, pts/mb->end_pts);

    int use_zyklus=0;

    fill_from_cache(ctx, NULL, &in_cidx, NULL, mb->start_y+scale*(-mb->h/2-0.5), scale);

    tmp_cidx= in_cidx;

    memset(color, 0, sizeof(*color)*mb->w);

    for(y=0; y<mb->h; y++){

        int y1= y+1;

        const double ci=mb->start_y+scale*(y-mb->h/2);

        fill_from_cache(ctx, NULL, &in_cidx, &next_cidx, ci, scale);

        if(y1<mb->h){

            memset(color+linesize*y1, 0, sizeof(*color)*mb->w);

            fill_from_cache(ctx, color+linesize*y1, &tmp_cidx, NULL, ci + 3*scale/2, scale);

        }



        for(x=0; x<mb->w; x++){

            float epsilon;

            const double cr=mb->start_x+scale*(x-mb->w/2);

            double zr=cr;

            double zi=ci;

            uint32_t c=0;

            double dv= mb->dither / (double)(1LL<<32);

            mb->dither= mb->dither*1664525+1013904223;



            if(color[x + y*linesize] & 0xFF000000)

                continue;

            if(interpol(mb, color, x, y, linesize)){

                if(next_cidx < mb->cache_allocated){

                    mb->next_cache[next_cidx  ].p[0]= cr;

                    mb->next_cache[next_cidx  ].p[1]= ci;

                    mb->next_cache[next_cidx++].val = color[x + y*linesize];

                }

                continue;

            }



            use_zyklus= (x==0 || mb->inner!=BLACK ||color[x-1 + y*linesize] == 0xFF000000);

            if(use_zyklus)

                epsilon= scale*1*sqrt(SQR(x-mb->w/2) + SQR(y-mb->h/2))/mb->w;



#define Z_Z2_C(outr,outi,inr,ini)\

            outr= inr*inr - ini*ini + cr;\

            outi= 2*inr*ini + ci;



#define Z_Z2_C_ZYKLUS(outr,outi,inr,ini, Z)\

            Z_Z2_C(outr,outi,inr,ini)\

            if(use_zyklus){\

                if(Z && fabs(mb->zyklus[i>>1][0]-outr)+fabs(mb->zyklus[i>>1][1]-outi) <= epsilon)\

                    break;\

            }\

            mb->zyklus[i][0]= outr;\

            mb->zyklus[i][1]= outi;\







            for(i=0; i<mb->maxiter-8; i++){

                double t;

                Z_Z2_C_ZYKLUS(t, zi, zr, zi, 0)

                i++;

                Z_Z2_C_ZYKLUS(zr, zi, t, zi, 1)

                i++;

                Z_Z2_C_ZYKLUS(t, zi, zr, zi, 0)

                i++;

                Z_Z2_C_ZYKLUS(zr, zi, t, zi, 1)

                i++;

                Z_Z2_C_ZYKLUS(t, zi, zr, zi, 0)

                i++;

                Z_Z2_C_ZYKLUS(zr, zi, t, zi, 1)

                i++;

                Z_Z2_C_ZYKLUS(t, zi, zr, zi, 0)

                i++;

                Z_Z2_C_ZYKLUS(zr, zi, t, zi, 1)

                if(zr*zr + zi*zi > mb->bailout){

                    i-= FFMIN(7, i);

                    for(; i<mb->maxiter; i++){

                        zr= mb->zyklus[i][0];

                        zi= mb->zyklus[i][1];

                        if(zr*zr + zi*zi > mb->bailout){

                            switch(mb->outer){

                            case            ITERATION_COUNT: zr = i; break;

                            case NORMALIZED_ITERATION_COUNT: zr= i + log2(log(mb->bailout) / log(zr*zr + zi*zi)); break;

                            }

                            c= lrintf((sin(zr)+1)*127) + lrintf((sin(zr/1.234)+1)*127)*256*256 + lrintf((sin(zr/100)+1)*127)*256;

                            break;

                        }

                    }

                    break;

                }

            }

            if(!c){

                if(mb->inner==PERIOD){

                int j;

                for(j=i-1; j; j--)

                    if(SQR(mb->zyklus[j][0]-zr) + SQR(mb->zyklus[j][1]-zi) < epsilon*epsilon*10)

                        break;

                if(j){

                    c= i-j;

                    c= ((c<<5)&0xE0) + ((c<<16)&0xE000) + ((c<<27)&0xE00000);

                }

                }else if(mb->inner==CONVTIME){

                    c= floor(i*255.0/mb->maxiter+dv)*0x010101;

                } else if(mb->inner==MINCOL){

                    int j;

                    double closest=9999;

                    int closest_index=0;

                    for(j=i-1; j>=0; j--)

                        if(SQR(mb->zyklus[j][0]) + SQR(mb->zyklus[j][1]) < closest){

                            closest= SQR(mb->zyklus[j][0]) + SQR(mb->zyklus[j][1]);

                            closest_index= j;

                        }

                    closest = sqrt(closest);

                    c= lrintf((mb->zyklus[closest_index][0]/closest+1)*127+dv) + lrintf((mb->zyklus[closest_index][1]/closest+1)*127+dv)*256;

                }

            }

            c |= 0xFF000000;

            color[x + y*linesize]= c;

            if(next_cidx < mb->cache_allocated){

                mb->next_cache[next_cidx  ].p[0]= cr;

                mb->next_cache[next_cidx  ].p[1]= ci;

                mb->next_cache[next_cidx++].val = c;

            }

        }

        fill_from_cache(ctx, NULL, &in_cidx, &next_cidx, ci + scale/2, scale);

    }

    FFSWAP(void*, mb->next_cache, mb->point_cache);

    mb->cache_used = next_cidx;

    if(mb->cache_used == mb->cache_allocated)

        av_log(0, AV_LOG_INFO, "Mandelbrot cache is too small!\n");

}
