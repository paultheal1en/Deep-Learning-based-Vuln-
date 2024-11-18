static int get_max_p_order(int max_porder, int n, int order)

{

    int porder, max_parts;



    porder = max_porder;

    while(porder > 0) {

        max_parts = (1 << porder);

        if(!(n % max_parts) && (n > max_parts*order)) {

            break;

        }

        porder--;

    }

    return porder;

}
