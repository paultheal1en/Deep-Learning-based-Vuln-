static void dump(unsigned char *buf,size_t len)

{

	int i;

	for(i=0;i<len;i++) {

		if ((i&15)==0) printf("%04x  ",i);

		printf("%02x ",buf[i]);

		if ((i&15)==15) printf("\n");

	}

	printf("\n");

}
