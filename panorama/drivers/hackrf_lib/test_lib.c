#include <stdio.h>
#include "hq_api.h"

int main() {
    HqStatus status;
    int r = hq_get_status(&status);
    printf("Library test: get_status returned %d\n", r);
    return 0;
}
