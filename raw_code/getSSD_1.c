static uint64_t getSSD(uint8_t *src1, uint8_t *src2, int stride1, int stride2, int w, int h){

	int x,y;

	uint64_t ssd=0;



//printf("%d %d\n", w, h);



	for(y=0; y<h; y++){

		for(x=0; x<w; x++){

			int d= src1[x + y*stride1] - src2[x + y*stride2];

			ssd+= d*d;

//printf("%d", abs(src1[x + y*stride1] - src2[x + y*stride2])/26 );

		}

//printf("\n");

	}

	return ssd;

}
