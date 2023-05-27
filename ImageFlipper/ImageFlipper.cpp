#include "../Output/Picture.hpp"
#include <iostream>
#include <sstream>
#include <vector>
#include <filesystem>

void splitImages(const std::string& path) {
	static constexpr int BORDER_WIDTH = 10;
	static constexpr int ROWS = 2;
	static constexpr int COLS = 3;
	static const std::string COL_NAMES[COLS] = { "p1", "0", "m1" };
	static const std::string ROW_NAMES[ROWS] = { "side", "top" };

	for (const auto& entry : std::filesystem::directory_iterator(path)) {
		std::string inFile = entry.path().string();
		std::cout << inFile << std::endl;
		if (inFile.substr(inFile.length() - 4, 4) == ".bmp") {
			Picture inImage;
			inImage.load(inFile, false);
			int width = inImage.getWidth() / COLS;
			int height = inImage.getHeight() / ROWS;
			Picture outImage(width, height);
			for (int row = 0; row < ROWS; ++row) {
				for (int col = 0; col < COLS; ++col) {
					for (int x = BORDER_WIDTH; x < width - BORDER_WIDTH; ++x) {
						for (int y = BORDER_WIDTH; y < height - BORDER_WIDTH; ++y) {
							bool in_borders = 1 < y && 1 < x && y < height - 1 && x < width - 1;
							int src_x = col * width + x;
							int src_y = row * height + y;
							outImage.setColor(x, y, inImage.getColor(src_x, src_y));
						}
					}
					std::string outFile = "results/" + inFile.substr(0, inFile.length() - 4) + "_" + ROW_NAMES[row] + "_" + COL_NAMES[col] + ".bmp";
					outImage.save(outFile, false);
					std::cout << "Done flipping: " << outFile << std::endl;
				}
			}
		}
	}
}

void flipImages(const std::string& path) {
	for (const auto& entry : std::filesystem::directory_iterator(path)) {
		std::string filename = entry.path().string();
		if (entry.is_directory()) {
			flipImages(filename);
		}
		else if (filename.substr(filename.length() - 4, 4) == ".bmp") {
			static constexpr int ROWS = 2;
			static constexpr int COLS = 3;
			Picture oldImage;
			oldImage.load(filename, false);
			int width = oldImage.getWidth() / COLS;
			int height = oldImage.getHeight() / ROWS;
			Picture newImage(width * COLS, height * ROWS);
			for (int row = 0; row < ROWS; ++row) {
				for (int col = 0; col < COLS; ++col) {
					for (int x = 0; x < width; ++x) {
						for (int y = 0; y < height; ++y) {
							bool in_borders = 0 < y && 0 < x && y < height - 1 && x < width - 1;
							int src_x = col * width + x;
							int src_y = row * height + y;
							int dst_x = in_borders ? col * width + (width - x - 1) : src_x;
							int dst_y = src_y;
							if (row == (ROWS - 1) && in_borders)
							{
								dst_y = row * height + (height - y - 1);
							}
							newImage.setColor(dst_x, dst_y, oldImage.getColor(src_x, src_y));
						}
					}
				}
			}
			newImage.save(filename, false);
			std::cout << "Done flipping: " << filename << std::endl;
		}
	}
}

int main( int argc, char** argv )
{
	std::string directory = "splitting";
	//flipImages(directory);
	splitImages(directory);
	return 0;
}
