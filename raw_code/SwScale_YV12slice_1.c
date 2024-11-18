void SwScale_YV12slice(unsigned char* srcptr[],int stride[], int srcSliceY ,

			     int srcSliceH, uint8_t* dstptr[], int dststride, int dstbpp,

			     int srcW, int srcH, int dstW, int dstH){



#ifdef RUNTIME_CPUDETECT

#ifdef CAN_COMPILE_X86_ASM

	// ordered per speed fasterst first

	if(gCpuCaps.hasMMX2)

		SwScale_YV12slice_MMX2(srcptr, stride, srcSliceY, srcSliceH, dstptr, dststride, dstbpp, srcW, srcH, dstW, dstH);

	else if(gCpuCaps.has3DNow)

		SwScale_YV12slice_3DNow(srcptr, stride, srcSliceY, srcSliceH, dstptr, dststride, dstbpp, srcW, srcH, dstW, dstH);

	else if(gCpuCaps.hasMMX)

		SwScale_YV12slice_MMX(srcptr, stride, srcSliceY, srcSliceH, dstptr, dststride, dstbpp, srcW, srcH, dstW, dstH);

	else

		SwScale_YV12slice_C(srcptr, stride, srcSliceY, srcSliceH, dstptr, dststride, dstbpp, srcW, srcH, dstW, dstH);

#else

		SwScale_YV12slice_C(srcptr, stride, srcSliceY, srcSliceH, dstptr, dststride, dstbpp, srcW, srcH, dstW, dstH);

#endif

#else //RUNTIME_CPUDETECT

#ifdef HAVE_MMX2

		SwScale_YV12slice_MMX2(srcptr, stride, srcSliceY, srcSliceH, dstptr, dststride, dstbpp, srcW, srcH, dstW, dstH);

#elif defined (HAVE_3DNOW)

		SwScale_YV12slice_3DNow(srcptr, stride, srcSliceY, srcSliceH, dstptr, dststride, dstbpp, srcW, srcH, dstW, dstH);

#elif defined (HAVE_MMX)

		SwScale_YV12slice_MMX(srcptr, stride, srcSliceY, srcSliceH, dstptr, dststride, dstbpp, srcW, srcH, dstW, dstH);

#else

		SwScale_YV12slice_C(srcptr, stride, srcSliceY, srcSliceH, dstptr, dststride, dstbpp, srcW, srcH, dstW, dstH);

#endif

#endif //!RUNTIME_CPUDETECT



}
