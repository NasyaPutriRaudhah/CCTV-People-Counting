"""
Data Persistence for Line Crossing Counts
"""
import os
import json
from datetime import datetime
from config import DATA_SAVE_FILE, STARTING_COUNT


class DataPersistence:
    """Handle saving and loading of counting data"""
    
    def __init__(self, filename=DATA_SAVE_FILE):
        self.filename = filename
        self.data = {
            'starting_count': STARTING_COUNT,
            'current_count': STARTING_COUNT,
            'total_entries': 0,
            'total_exits': 0,
            'session_start': None,
            'last_update': None,
            'history': []  # Log of all crossing events
        }
        self.load_data()
    
    def load_data(self):
        """Load existing data if available"""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    loaded_data = json.load(f)
                    # Keep the loaded data but update starting count if changed in config
                    if loaded_data.get('starting_count') == STARTING_COUNT:
                        self.data = loaded_data
                        print(f"✓ Loaded existing data from {self.filename}")
                        print(f"  Current count: {self.data['current_count']}")
                        print(f"  Total entries: {self.data['total_entries']}")
                        print(f"  Total exits: {self.data['total_exits']}")
                    else:
                        print(f"⚠ Starting count changed, resetting data")
                        self.data['starting_count'] = STARTING_COUNT
                        self.data['current_count'] = STARTING_COUNT
                        self.save_data()
            except Exception as e:
                print(f"⚠ Error loading data: {e}")
                self.save_data()
        else:
            print(f"✓ Starting fresh with count: {STARTING_COUNT}")
            self.data['session_start'] = datetime.now().isoformat()
            self.save_data()
    
    def save_data(self):
        """Save current data to file"""
        self.data['last_update'] = datetime.now().isoformat()
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"⚠ Error saving data: {e}")
    
    def add_entries(self, count):
        """Add entry events"""
        self.data['current_count'] += count
        self.data['total_entries'] += count
        
        # Log events
        for _ in range(count):
            event = {
                'timestamp': datetime.now().isoformat(),
                'type': 'entry',
                'delta': 1,
                'count_after': self.data['current_count']
            }
            self.data['history'].append(event)
        
        # Keep only last 1000 events
        if len(self.data['history']) > 1000:
            self.data['history'] = self.data['history'][-1000:]
    
    def add_exits(self, count):
        """Add exit events"""
        self.data['current_count'] -= count
        self.data['total_exits'] += count
        
        # Log events
        for _ in range(count):
            event = {
                'timestamp': datetime.now().isoformat(),
                'type': 'exit',
                'delta': -1,
                'count_after': self.data['current_count']
            }
            self.data['history'].append(event)
        
        # Keep only last 1000 events
        if len(self.data['history']) > 1000:
            self.data['history'] = self.data['history'][-1000:]
    
    def get_current_count(self):
        """Get current count"""
        return self.data['current_count']
    
    def get_summary(self):
        """Get summary statistics"""
        return {
            'starting_count': self.data['starting_count'],
            'current_count': self.data['current_count'],
            'total_entries': self.data['total_entries'],
            'total_exits': self.data['total_exits'],
            'net_change': self.data['current_count'] - self.data['starting_count'],
            'session_start': self.data['session_start'],
            'last_update': self.data['last_update']
        }