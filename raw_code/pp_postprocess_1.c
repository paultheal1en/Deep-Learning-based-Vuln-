void  pp_postprocess(const uint8_t * src[3], const int srcStride[3],

                     uint8_t * dst[3], const int dstStride[3],

                     int width, int height,

                     const QP_STORE_T *QP_store,  int QPStride,

                     pp_mode *vm,  void *vc, int pict_type)

{

    int mbWidth = (width+15)>>4;

    int mbHeight= (height+15)>>4;

    PPMode *mode = vm;

    PPContext *c = vc;

    int minStride= FFMAX(FFABS(srcStride[0]), FFABS(dstStride[0]));

    int absQPStride = FFABS(QPStride);



    // c->stride and c->QPStride are always positive

    if(c->stride < minStride || c->qpStride < absQPStride)

        reallocBuffers(c, width, height,

                       FFMAX(minStride, c->stride),

                       FFMAX(c->qpStride, absQPStride));



    if(!QP_store || (mode->lumMode & FORCE_QUANT)){

        int i;

        QP_store= c->forcedQPTable;

        absQPStride = QPStride = 0;

        if(mode->lumMode & FORCE_QUANT)

            for(i=0; i<mbWidth; i++) c->forcedQPTable[i]= mode->forcedQuant;

        else

            for(i=0; i<mbWidth; i++) c->forcedQPTable[i]= 1;

    }



    if(pict_type & PP_PICT_TYPE_QP2){

        int i;

        const int count= FFMAX(mbHeight * absQPStride, mbWidth);

        for(i=0; i<(count>>2); i++){

            ((uint32_t*)c->stdQPTable)[i] = (((const uint32_t*)QP_store)[i]>>1) & 0x7F7F7F7F;

        }

        for(i<<=2; i<count; i++){

            c->stdQPTable[i] = QP_store[i]>>1;

        }

        QP_store= c->stdQPTable;

        QPStride= absQPStride;

    }



    if(0){

        int x,y;

        for(y=0; y<mbHeight; y++){

            for(x=0; x<mbWidth; x++){

                av_log(c, AV_LOG_INFO, "%2d ", QP_store[x + y*QPStride]);

            }

            av_log(c, AV_LOG_INFO, "\n");

        }

        av_log(c, AV_LOG_INFO, "\n");

    }



    if((pict_type&7)!=3){

        if (QPStride >= 0){

            int i;

            const int count= FFMAX(mbHeight * QPStride, mbWidth);

            for(i=0; i<(count>>2); i++){

                ((uint32_t*)c->nonBQPTable)[i] = ((const uint32_t*)QP_store)[i] & 0x3F3F3F3F;

            }

            for(i<<=2; i<count; i++){

                c->nonBQPTable[i] = QP_store[i] & 0x3F;

            }

        } else {

            int i,j;

            for(i=0; i<mbHeight; i++) {

                for(j=0; j<absQPStride; j++) {

                    c->nonBQPTable[i*absQPStride+j] = QP_store[i*QPStride+j] & 0x3F;

                }

            }

        }

    }



    av_log(c, AV_LOG_DEBUG, "using npp filters 0x%X/0x%X\n",

           mode->lumMode, mode->chromMode);



    postProcess(src[0], srcStride[0], dst[0], dstStride[0],

                width, height, QP_store, QPStride, 0, mode, c);



    if (!(src[1] && src[2] && dst[1] && dst[2]))

        return;



    width  = (width )>>c->hChromaSubSample;

    height = (height)>>c->vChromaSubSample;



    if(mode->chromMode){

        postProcess(src[1], srcStride[1], dst[1], dstStride[1],

                    width, height, QP_store, QPStride, 1, mode, c);

        postProcess(src[2], srcStride[2], dst[2], dstStride[2],

                    width, height, QP_store, QPStride, 2, mode, c);

    }

    else if(srcStride[1] == dstStride[1] && srcStride[2] == dstStride[2]){

        linecpy(dst[1], src[1], height, srcStride[1]);

        linecpy(dst[2], src[2], height, srcStride[2]);

    }else{

        int y;

        for(y=0; y<height; y++){

            memcpy(&(dst[1][y*dstStride[1]]), &(src[1][y*srcStride[1]]), width);

            memcpy(&(dst[2][y*dstStride[2]]), &(src[2][y*srcStride[2]]), width);

        }

    }

}
