static uint16_t roundToInt16(int64_t f){

	int r= (f + (1<<15))>>16;

	     if(r<-0x7FFF) return 0x8000;

	else if(r> 0x7FFF) return 0x7FFF;

	else               return r;

}
