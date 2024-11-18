void *av_fast_realloc(void *ptr, unsigned int *size, unsigned int min_size)

{

    if(min_size < *size) 

        return ptr;

    

    *size= 17*min_size/16 + 32;



    return av_realloc(ptr, *size);

}
