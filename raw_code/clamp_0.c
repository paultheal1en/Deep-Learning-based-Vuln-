static inline int clamp(int value, int min, int max)

{

   if (value < min)

       return min;

   else if (value > max)

       return max;

   else 

       return value;

}
