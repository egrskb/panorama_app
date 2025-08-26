#include <stdio.h>
#include <stdlib.h>
#include "hackrf_master.h"

int main() {
    printf("Тестирование HackRF Master библиотеки...\n");
    
    // Проверяем, что константы определены
    printf("MAX_SEGMENTS: %d\n", MAX_SEGMENTS);
    printf("DEFAULT_FFT_SIZE: %d\n", DEFAULT_FFT_SIZE);
    printf("MAX_CALIBRATION_ENTRIES: %d\n", MAX_CALIBRATION_ENTRIES);
    
    // Проверяем функции
    printf("hq_device_count(): %d\n", hq_device_count());
    printf("hq_get_segment_mode(): %d\n", hq_get_segment_mode());
    
    printf("Тест завершен успешно!\n");
    return 0;
}
