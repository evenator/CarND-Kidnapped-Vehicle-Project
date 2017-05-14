/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	std::normal_distribution<double> x_dist(x, std[0]);
	std::normal_distribution<double> y_dist(y, std[1]);
	std::normal_distribution<double> theta_dist(theta, std[2]);
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	num_particles = 10;
	particles.clear();
	for (size_t i; i < num_particles; ++i) {
		particles.emplace_back();
		particles.back().id = i;
		particles.back().x = x_dist(generator);
		particles.back().y = y_dist(generator);
		particles.back().theta = theta_dist(generator);
		while (particles.back().theta < -M_PI)
			particles.back().theta += M_PI;
		while (particles.back().theta > M_PI)
			particles.back().theta -= M_PI;
		particles.back().weight = 1.0;
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
		double velocity, double yaw_rate) {
	std::normal_distribution<double> x_dist(0, std_pos[0]);
	std::normal_distribution<double> y_dist(0, std_pos[1]);
	std::normal_distribution<double> theta_dist(0, std_pos[2]);
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	double d_theta = yaw_rate * delta_t;
	double d_pos = velocity * delta_t;
	for (std::vector<Particle>::iterator p = particles.begin();
			p != particles.end(); ++p) {
		if (yaw_rate == 0) {
			p->x += d_pos * cos(p->theta);
			p->y += d_pos * sin(p->theta);
		} else {
			p->x += velocity / yaw_rate
					* (sin(p->theta + d_theta) - sin(p->theta));
			p->y += velocity / yaw_rate
					* (cos(p->theta) - cos(p->theta + d_theta));
		}
		p->theta += d_theta;
		p->theta += theta_dist(generator);
		p->theta = atan2(sin(p->theta), cos(p->theta));
		p->x += x_dist(generator);
		p->y += y_dist(generator);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
		std::vector<LandmarkObs>& observations) {
	for (std::vector<LandmarkObs>::iterator obs = observations.begin();
			obs != observations.end(); ++obs) {
		double closest_dist = std::numeric_limits<double>::max();
		for (size_t i = 0; i < predicted.size(); ++i) {
			LandmarkObs const& pred = predicted[i];
			double this_dist = dist(pred.x, pred.y, obs->x, obs->y);
			if (this_dist < closest_dist) {
				closest_dist = this_dist;
				obs->id = i;
			}
		}
	}
}

/**
 * Calculate the probability of (x,y), given a normal distribution with mean_x, mean_y and std_x, std_y
 */
double gaussianProb(double x, double y, double mean_x, double mean_y,
		double std_x, double std_y) {
	double dx = x - mean_x;
	double dy = y - mean_y;
	double var_x = std_x * std_x;
	double var_y = std_y * std_y;
	double p = (1.0 / (2 * M_PI * std_x * std_y))
			* exp(-(dx * dx / (2 * var_x) + dy * dy / (2 * var_y)));
	return p;
}

std::vector<LandmarkObs> predictObservations(Particle const& pose,
		Map const& map) {
	std::vector<LandmarkObs> pred;
	typedef Map::single_landmark_s Landmark;
	typedef std::vector<Landmark> LandmarkVector;
	for (LandmarkVector::const_iterator landmark = map.landmark_list.begin();
			landmark != map.landmark_list.end(); ++landmark) {
		pred.emplace_back();
		pred.back().id = landmark->id_i;
		double x_trans = landmark->x_f - pose.x;
		double y_trans = landmark->y_f - pose.y;
		double r = sqrt(x_trans * x_trans + y_trans * y_trans);
		double bearing = atan2(y_trans, x_trans);

		pred.back().x = r * cos(bearing - pose.theta);
		pred.back().y = r * sin(bearing - pose.theta);
	}
	return pred;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	for (std::vector<Particle>::iterator p = particles.begin();
			p != particles.end(); ++p) {
		std::vector<LandmarkObs> predictions = predictObservations(*p,
				map_landmarks);
		dataAssociation(predictions, observations);
		p->weight = 1;
		for (std::vector<LandmarkObs>::iterator obs = observations.begin();
				obs != observations.end(); ++obs) {
			p->weight *= gaussianProb(obs->x, obs->y, predictions[obs->id].x,
					predictions[obs->id].y, std_landmark[0], std_landmark[1]);
		}
	}
	weights.resize(particles.size(), 0);
	for (size_t i = 0; i < particles.size(); ++i) {
		weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample() {
	// Find the max weight
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	double max_weight = 0;
	for (std::vector<Particle>::const_iterator p = particles.begin();
			p != particles.end(); ++p) {
		if (p->weight > max_weight) {
			max_weight = p->weight;
		}
	}
	// Create new particles
	std::vector<Particle> new_particles;
	size_t index = std::uniform_int_distribution<>(0, particles.size()-1)(
			generator);
	double beta = 0;
	std::uniform_real_distribution<> dist(0, 2 * max_weight);
	for (size_t i = 0; i < particles.size(); ++i) {
		beta += dist(generator);
		while (weights[index] < beta) {
			beta -= weights[index];
			index = (index + 1) % particles.size();
		}
		new_particles.push_back(particles[index]);
	}
	particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " "
				<< particles[i].theta << "\n";
	}
	dataFile.close();
}
