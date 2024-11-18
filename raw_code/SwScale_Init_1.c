void SwScale_Init(){

    // generating tables:

    int i;

    for(i=0; i<768; i++){

	int c= MIN(MAX(i-256, 0), 255);

	clip_table[i]=c;

	yuvtab_2568[c]= clip_yuvtab_2568[i]=(0x2568*(c-16))+(256<<13);

	yuvtab_3343[c]= clip_yuvtab_3343[i]=0x3343*(c-128);

	yuvtab_0c92[c]= clip_yuvtab_0c92[i]=-0x0c92*(c-128);

	yuvtab_1a1e[c]= clip_yuvtab_1a1e[i]=-0x1a1e*(c-128);

	yuvtab_40cf[c]= clip_yuvtab_40cf[i]=0x40cf*(c-128);

    }



    for(i=0; i<768; i++)

    {

    	int v= clip_table[i];

	clip_table16b[i]= v>>3;

	clip_table16g[i]= (v<<3)&0x07E0;

	clip_table16r[i]= (v<<8)&0xF800;

	clip_table15b[i]= v>>3;

	clip_table15g[i]= (v<<2)&0x03E0;

	clip_table15r[i]= (v<<7)&0x7C00;

    }



}
