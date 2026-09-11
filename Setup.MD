# List of commands

1. Connect the glasses to the phone app and record. https://facebookresearch.github.io/projectaria_tools/gen2/ark/companion-app/start
2. Connect the smart glasses to the PC using USB C
3. ```bash
   conda create -n aria python=3.11
   ```
4. ```bash
   conda activate aria
   ```
5. ```bash
   pip install projectaria-client-sdk
   ```
6. ```bash
   sudo apt install adb fastboot
   ```
7. ```bash
   aria_gen2 device list
   ```
8. ```bash
   aria_gen2 auth pair
   ```
9. Open the mobile phone app and authenticate
10. ```bash
    aria_gen2 recording list
    ```
11. ```bash
    aria_gen2 recording download -u 2c7b95d7-3534-43c7-bfe4-7db16a7ffa3d -o ./
    ```
12. ```bash
    aria_rerun_viewer --vrs videoName.vrs
    ```
