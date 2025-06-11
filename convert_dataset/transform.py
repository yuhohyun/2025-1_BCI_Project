import os
import sys
import numpy as np
import scipy.io as sio
import mne
import glob

current_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(current_path)[0]
sys.path.append(current_path)
sys.path.append(root_path)

class LoadData:
    def __init__(self, eeg_file_path: str):
        self.eeg_file_path = eeg_file_path
        self.raw_eeg_subject = None

    def load_raw_data_gdf(self, file_to_load):
        self.raw_eeg_subject = mne.io.read_raw_gdf(self.eeg_file_path + '/' + file_to_load)
        return self

    def load_raw_data_mat(self,file_to_load):
        self.raw_eeg_subject = sio.loadmat(self.eeg_file_path + '/' + file_to_load)

    def get_all_files(self, file_path_extension: str = None):
        if file_path_extension:
            return glob.glob(self.eeg_file_path+'/'+file_path_extension)
        return os.listdir(self.eeg_file_path)

class LoadBCIC(LoadData):
    """Subclass of LoadData for loading BCI Competition IV Dataset 2a"""
    def __init__(self, file_to_load, *args):
        self.stimcodes = ['769', '770', '771', '772']
        # self.epoched_data={}
        self.file_to_load = file_to_load
        self.channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']
        self.fs = None
        super(LoadBCIC, self).__init__(*args)

    def get_epochs(self, tmin=-4.5, tmax=5.0, baseline=None,reject = False):
        
        self.load_raw_data_gdf(self.file_to_load)
        raw_data = self.raw_eeg_subject
        events, event_ids = mne.events_from_annotations(raw_data)
        self.fs = raw_data.info.get('sfreq')
        if reject == True:
            reject_events = mne.pick_events(events,[1])
            reject_oneset = reject_events[:,0]/self.fs
            duration = [4]*len(reject_events)
            descriptions = ['bad trial']*len(reject_events)
            blink_annot = mne.Annotations(reject_oneset,duration,descriptions)
            raw_data.set_annotations(blink_annot)
        
        stims =[value for key, value in event_ids.items() if key in self.stimcodes]
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                            baseline=baseline, preload=True, proj=False, reject_by_annotation=True)
        epochs = epochs.drop_channels(self.channels_to_remove)
        self.y_labels = epochs.events[:, -1] - min(epochs.events[:, -1])
        self.x_data = epochs.get_data()*1e6
        # length = len(self.x_data)
        eeg_data={'x_data': self.x_data,
                  'y_labels': self.y_labels,
                  'fs': self.fs
                  }
        return eeg_data
    
class LoadBCIC_E(LoadData):
    """A class to lode the test data of the BICI IV 2a dataset"""
    def __init__(self, file_to_load, lable_name, *args):
        self.stimcodes = ('783')
        # self.epoched_data={}
        self.label_name = lable_name # the path of the test label
        self.file_to_load = file_to_load
        self.channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']
        super(LoadBCIC_E, self).__init__(*args)

    def get_epochs(self, tmin=-4.5, tmax=5.0, baseline=None):
        self.load_raw_data_gdf(self.file_to_load)
        raw_data = self.raw_eeg_subject
        self.fs = raw_data.info.get('sfreq')
        events, event_ids = mne.events_from_annotations(raw_data)
        stims =[value for key, value in event_ids.items() if key in self.stimcodes]
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                            baseline=baseline, preload=True, proj=False, reject_by_annotation=False)
        epochs = epochs.drop_channels(self.channels_to_remove)
        label_info  = sio.loadmat(os.path.join(self.eeg_file_path,self.label_name))
        #label_info shape:(288, 1)
        self.y_labels = label_info['classlabel'].reshape(-1) -1
        # print(self.y_labels)
        self.x_data = epochs.get_data()*1e6
        eeg_data={'x_data': self.x_data,
                  'y_labels': self.y_labels,
                  'fs': self.fs}
        return eeg_data


def Load_BCIC_2a_raw_data(data_path):
    '''
    Load the BCIC IV 2a dataset.
    '''
    
    for sub in range(1,10):
        data_name = r'A0{}T.gdf'.format(sub)
        data_loader = LoadBCIC(data_name, data_path)
        data = data_loader.get_epochs(tmin=0.5, tmax=4.5)
        
        # train_x = np.array(data['x_data'])[:, :, :1000]
        train_x = np.array(data['x_data'])[:, np.newaxis, :, :1000]  # shape: (batch, 1, channels, time)
        train_y = np.array(data['y_labels'])
        
        data_name = r'A0{}E.gdf'.format(sub)
        label_name = r'A0{}E.mat'.format(sub)
        data_loader = LoadBCIC_E(data_name, label_name, data_path)
        data = data_loader.get_epochs(tmin=0.5, tmax=4.5)
        
        # test_x = np.array(data['x_data'])[:, :, :1000]
        test_x = np.array(data['x_data'])[:, np.newaxis, :, :1000]
        test_y = data['y_labels']
        
        train_x = np.array(train_x)
        train_y = np.array(train_y).reshape(-1)
        # if ems:
        #     train_x = EMstandardize(train_x)
        
        test_x = np.array(test_x)
        test_y = np.array(test_y).reshape(-1)
        # if ems:
        #     test_x = EMstandardize(test_x)
        
        print('trian_x:',train_x.shape)
        print('train_y:',train_y.shape)
        
        print('test_x:',test_x.shape)
        print('test_y:',test_y.shape)
        
        SAVE_path = os.path.join(root_path,'dataset','BCICIV_2a') 

        if not os.path.exists(SAVE_path):
            os.makedirs(SAVE_path)
            
        SAVE_test = os.path.join(SAVE_path,r'sub{}_test'.format(sub))
        SAVE_train = os.path.join(SAVE_path,'sub{}_train'.format(sub))
        
        if not os.path.exists(SAVE_test):
            os.makedirs(SAVE_test)
        if not os.path.exists(SAVE_train):
            os.makedirs(SAVE_train)
            
        sio.savemat(os.path.join(SAVE_train, "Data.mat"), {'x_data': train_x,'y_data': train_y})
        sio.savemat(os.path.join(SAVE_test, "Data.mat"), {'x_data': test_x, 'y_data': test_y})
        print('Saved successfully!')

    
if __name__ == '__main__':

    data_path = os.path.join(root_path,'dataset','BCICIV_2a_gdf')
    Load_BCIC_2a_raw_data(data_path)